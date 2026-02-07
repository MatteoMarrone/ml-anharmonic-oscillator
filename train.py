# Imports
import torch, time
from torch import nn
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt # Plotting library
from tqdm import tqdm # Progress bar
from scipy.interpolate import interp1d
from datetime import datetime
from zoneinfo import ZoneInfo
import csv, os, json
import h5py

from parameters import * #we import all the variables/constants defined in that file
from model import HarmonicNQS
from perturbed_energy import pertTheory_Energy
from numerical_schrodinger import numericalEnergy_Schrodinger
from mc_chain import mh_samples
#-------------------------------------------
#This file corresponds to the main traning process of the NN, using the energy calculated via MC as the loss function
#-------------------------------------------
# Hardware (CPU or GPU)
dev = 'cpu' # can be changed to 'cuda' for GPU usage
device = torch.device(dev)
net = HarmonicNQS(W1, B, W2).to(device)
#---------------------------

save_chain = True
#whether we want to save the values of the markov chain in a file or not
if save_chain == False:
    print("-- save_chain = False | We will not save in a file the values of the Markov chain for this simulation.")

main_dir = "results"
os.makedirs(main_dir, exist_ok=True)

out_dir = f"(lambda={L:.3f})"
out_dir = os.path.join(main_dir, out_dir)
print(f"folder name: '{out_dir}'")
if os.path.exists(out_dir) and os.listdir(out_dir):
    ans = input(f'Folder "{out_dir}" already exists and is not empty. Overwrite? [y/N]: ')
    if ans.lower() != 'y':
        print("Aborting.")
        exit()
else:
    os.makedirs(out_dir, exist_ok=True)

if save_chain:
    h5file = h5py.File(f"{out_dir}/chains.h5", "w")
    # Create expandable dataset
    chain_dset = h5file.create_dataset(
        "markov_chain",
        shape=(0, n_samples, 1),
        maxshape=(None, n_samples, 1),
        dtype='float32',
        compression="lzf",
        chunks=(1, n_samples, 1)
    )

#General parameters of the NN
#--------------------------------
torch.manual_seed(seed)

h = (train_b - train_a)/(Nx - 1)  # Mesh parameter "h"

wi = torch.empty(Nx, 1).fill_(h).to(device) 
Q_train = torch.linspace(train_a, train_b, Nx, requires_grad=True, device=device)
target = (1/np.pi)**(1/4) * torch.exp(-Q_train.pow(2)/2).to(device) #True ground state of the system.
    #.pow(2) takes the square of the 1d tensor Q_train

# -- Physical model values
print("-----")
print("lambda = ", L)
pt_energy, pt_norder, pt_denergy = pertTheory_Energy(L)

if L != 0:
    print(f"Energy in PT = {pt_energy:.4f} +- {pt_denergy:.4f} at order n={pertTheory_Energy(L)[1]}")
else:
    print(f"Energy in PT = {pertTheory_Energy(L)[0]:.4f}")

print(f"Numerical energy: {numericalEnergy_Schrodinger(L):.4f}\n")

#-----------------------------------------------

X = Q_train.clone().unsqueeze(1) # Training set
X_det = X.clone().detach() # Training set without gradients

def V(x):
    """
    potential energy for an anharmonic oscillator
    x is a torch tensor
    """
    return 0.5 * x.pow(2) + L*x.pow(4)

def mc_energy(net, x_samples, eps=0):
    """
    Computes MC estimate of the energy and returns a scalar loss.
    """
    x_samples.requires_grad_(True)

    psi = net(x_samples)

    dpsi_dx = torch.autograd.grad(
        psi, x_samples,
        grad_outputs=torch.ones_like(psi),
        create_graph=True
    )[0]

    d2psi_dx2 = torch.autograd.grad(
        dpsi_dx, x_samples,
        grad_outputs=torch.ones_like(dpsi_dx),
        create_graph=True
    )[0]

    kinetic = -0.5 * d2psi_dx2 / (psi + eps)
    # kinetic = 0.5 * (dpsi_dx/(psi+eps)).pow(2) #integrated by parts formula
    potential = V(x_samples)

    E_loc = kinetic + potential
    
    return kinetic, potential, E_loc

def manual_step(net, x_samples, E_loc, lr=lr, eps=0):
    """
    Perform ONE manual update of the parameters following the descent of the loss function E_loc.
    Correct formula for VMC gradient: d<E>/dθ = 2 < (O - <O>) (E_loc - <E>) >
    """

    params = list(net.parameters())
    N = x_samples.shape[0]  # length of the chain

    # --- Build O matrix ---
    O_list = []

    for k in range(N):
        net.zero_grad()

        psi = net(x_samples[k:k+1])
        log_psi = torch.log(psi + eps)

        grads = torch.autograd.grad(
            log_psi,
            params,
            # retain_graph=True
        )

        O_k = torch.cat([g.flatten() for g in grads])
        O_list.append(O_k)

    O = torch.stack(O_list)  # (N_samples, n_params)
    
    # --- Centered O and E_loc ---
    E_loc = E_loc.squeeze()
    E_mean = E_loc.mean()
    O_mean = O.mean(dim=0)

    # VMC gradient (centered)
    dE = 2 * ((O - O_mean) * (E_loc.unsqueeze(1) - E_mean)).mean(dim=0)

    # --- Update parameters manually ---
    idx = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p -= lr * dE[idx:idx+n].view_as(p)
            idx += n


def flatten_params(net):
    return torch.cat([p.detach().view(-1) for p in net.parameters()])

#----------------------
loss_accum = []
dE_accum = []
K_accum = []
U_accum = []

#WE CREATE THE DATA FILES
# --- CSV for scalar data
energy_file = open(f"{out_dir}/energies.csv", "w", newline="")
energy_writer = csv.writer(energy_file)
energy_writer.writerow(["step", "K", "U", "E", "dE"])

# Create metadata dictionary
metadata = {
    # Markov chain
    "n_samples": n_samples,
    "sigma": sigma, #of the proposal distribution
    "skip_size": skip_size,
    "burn_in": burn_in,
    
    # Neural network
    "Nin": Nin,
    "Nout": Nout,
    "Nhid": Nhid,
    
    # Training
    "epochs": epochs,
    "lr": lr,
    
    # Mesh
    "Nx": Nx,
    "train_a": train_a,
    "train_b": train_b,
    
    # Seed
    "seed": seed,

    #Lambda, anharmonic oscillator
    "lambda": L,
    "pt_energy": pt_energy,
    "pt_error_energy":pt_denergy,
    "pt_norder":int(pt_norder) if L!=0 else None

}

with open(os.path.join(out_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

# --- memory-mapped wavefunctions ---
psi_nn_mm = np.lib.format.open_memmap(
    f"{out_dir}/psi_nn.npy",
    mode="w+",
    dtype=np.float64,
    shape=(epochs, Nx, 1)
)

psi_norm_mm = np.lib.format.open_memmap(
    f"{out_dir}/psi_norm.npy",
    mode="w+",
    dtype=np.float64,
    shape=(epochs, Nx, 1)
)

np.save(f"{out_dir}/x_grid.npy", X_det.detach().cpu().numpy())

# ----------------------------
# Training loop
# ----------------------------
start_time = datetime.now(ZoneInfo("Europe/Madrid")).strftime("%H:%M:%S | %Y-%m-%d")
print(f"-- We will train the Neural Network... - Simulation started at: {start_time}")

try:
    # store parameters at previous step
    theta_prev = flatten_params(net)
    for i in tqdm(range(epochs), desc="Training the NQS...", disable=True):

        #1 Generate samples x - |psi|2 using MH
        x_chain, acc_ratio = mh_samples(net)

        #2 compute local energies
        K_arr, U_arr, E_loc = mc_energy(net, x_chain)

        #3 compute mean energy and error
        E_mean = E_loc.mean()
        dE_mean = E_loc.std()/np.sqrt(x_chain.shape[0])

        #4 update network
        print(f"- Epoch {i+1}")
        manual_step(net, x_chain, E_loc)

        #compute parameter displacement norm
        theta_new = flatten_params(net)
        delta_theta = torch.norm(theta_new - theta_prev).item()
        theta_prev = theta_new


        #5 print progress:
        print(f"Energy: {E_mean.item():.4f} +- {dE_mean.item():.4f}. Acc. ratio: {acc_ratio:.2f}. ||Δθ|| = {delta_theta:.3e}")
        print("------")

        #----
        psi_nn = net(X_det)
        psi2 = net(X_det).pow(2)
        N_norm = torch.tensordot(psi2, wi)
        psi_nn *= torch.sign(psi_nn[torch.argmax(torch.abs(psi_nn))]) #visual purposes
        psi_norm = psi_nn / torch.sqrt(N_norm)

        #we save the data
        energy_writer.writerow([
            i,
            K_arr.mean().item(),
            U_arr.mean().item(),
            E_mean.item(),
            dE_mean.item()
        ])
        
        energy_file.flush()

        psi_nn_mm[i] = psi_nn.detach().cpu().numpy()
        psi_norm_mm[i] = psi_norm.detach().cpu().numpy()

        # flush every step
        psi_nn_mm.flush()
        psi_norm_mm.flush()

        if save_chain == True and (i % SAVE_CHAIN_EVERY == 0):
            chain_np = x_chain.detach().cpu().numpy().astype(np.float32)
            chain_dset.resize(i+1, axis=0)
            chain_dset[i] = chain_np


except KeyboardInterrupt:
    print("\nTraining interrupted by user.")

finally:
    psi_nn_mm.flush()
    psi_norm_mm.flush()
    energy_file.close()
    h5file.close() if save_chain else None
    print("Data safely written to disk.")