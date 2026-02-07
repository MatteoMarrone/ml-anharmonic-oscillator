
# In this file we will test if the Markov chain parameters we are using are convenient.
import torch, time
from torch import nn
import numpy as np
import matplotlib.pyplot as plt # Plotting library
from tqdm import tqdm # Progress bar

from parameters import * #we import all the variables/constants defined in that file
from model import HarmonicNQS, HarmonicExact

run = False #whether we run the diagnostics of the chain or not

def mh_samples(net, x_init=0., n_samples=n_samples,eps=0, 
               sigma=sigma, skip_size=skip_size, 
               burn_in=burn_in, device='cpu',
               disable=True):
    """
    Generate samples x ~ |psi(x)|^2 using Metropolis-Hastings.
    We consider a Gaussian proposal distribution centered at the previous value and std = sigma
    """
    x = torch.tensor([[x_init]], device=device)

    psi2 = net(x).pow(2)

    n_store = n_samples + burn_in
    n_steps = n_store * skip_size

    # samples = []
    samples = torch.empty((n_store, 1), device=device)

    idx = 0
    accepted = 0
    for step in tqdm(range(n_steps),desc="-- MH Sampling --", disable=disable):
        # Propose move
        x_new = torch.normal(mean=x, std=sigma) #gaussian proposal distribution
        psi2_new = net(x_new).pow(2)

        # Metropolis acceptance
        A = psi2_new / (psi2 + eps)
        if torch.rand(1, device=device) < A:
            x = x_new
            psi2 = psi2_new
            accepted += 1

        if step % skip_size == 0:
            samples[idx] = x
            idx += 1
            
    return samples[burn_in:], accepted/n_steps

#----------------------------------------
if run:
    # -------------------------
    # Device
    # -------------------------
    device = "cpu"

    # -------------------------
    # Define / load wavefunction
    # -------------------------
    net = HarmonicNQS(W1,B,W2).to(device)
    net.eval()   # IMPORTANT: no training, fixed psi

    # -------------------------
    # Run MH sampling
    # -------------------------
    samples, acc_rate = mh_samples(
        net,
        x_init=0.0,
        n_samples=n_samples,
        sigma=sigma,
        skip_size=skip_size,
        burn_in=burn_in,
        device=device,
        disable=False
    )

    samples = samples.cpu().numpy().flatten()

    print("\n--- Metropolis-Hastings diagnostics ---")
    print(f"Number of samples : {len(samples)}")
    print(f"Acceptance rate : {acc_rate:.3f}")
    print(f"Sigma : {sigma}")
    print(f"Skip size : {skip_size}")
    print(f"Burn-in : {burn_in}")
    print(f"Mean point : {np.mean(samples)}")

    # -------------------------
    # Compute |psi(x)|^2 on grid
    # -------------------------
    x_grid = np.linspace(train_a, train_b, Nx)
    x_torch = torch.tensor(x_grid[:, None], dtype=torch.float32)

    with torch.no_grad():
        psi = net(x_torch).cpu().numpy().flatten()
        psi2 = psi**2

    # Normalize |psi|^2
    dx = x_grid[1] - x_grid[0]
    psi2 /= np.sum(psi2) * dx

    # -------------------------
    # Plot 1: Histogram vs |psi|^2
    # -------------------------
    plt.figure(figsize=(6, 4))

    plt.hist(
        samples,
        bins=80,
        density=True,
        alpha=0.5,
        label="MH samples"
    )

    plt.plot(
        x_grid,
        psi2,
        "k-",
        lw=2,
        label=r"$|\psi(x)|^2$"
    )

    plt.xlabel(r"$x$")
    plt.ylabel("Probability density")
    plt.title("Sampling of $|\psi(x)|^2$")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Plot 2: Samples vs index (scatter)
    # -------------------------
    plt.figure(figsize=(6, 4))

    idx = np.arange(len(samples))

    plt.scatter(
        idx,
        samples,
        s=12,        # marker size (small is better)
        alpha=0.6   # transparency helps for dense regions
    )

    plt.xlabel("Sample index")
    plt.ylabel(r"$x$")
    plt.title("Markov chain trace (scatter)")
    plt.tight_layout()
    plt.show()