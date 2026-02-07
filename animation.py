import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

from parameters import L
from perturbed_energy import pertTheory_Energy
from numerical_schrodinger import numericalEnergy_Schrodinger

animation = True
plot_last = True
show_plot = False

main_dir = "results"
# main_dir = "results_sr"
folder_name = f"(lambda={L:.3f})"
folder_name = os.path.join(main_dir, folder_name)
print(f"folder name: '{folder_name}'")

#animation and plot parameters
E0, n, dE0 = pertTheory_Energy(L) 
E0_true = numericalEnergy_Schrodinger(L)
n_sigmas = 1.5 #number of sigmas in energy plot y-axis limits
# ax_energy_lims = [E0-n_sigmas*dE0, E0+n_sigmas*dE0] if dE0!=0 else [0.4,0.6]
ax_energy_lims = [0.45,0.85]

fps = 20

# Load wavefunctions
psi_nn = np.load(f"{folder_name}/psi_nn.npy")
psi_norm = np.load(f"{folder_name}/psi_norm.npy")

# Load energies
energies = pd.read_csv(f"{folder_name}/energies.csv")

steps = energies["step"].values
E = energies["E"].values
U = energies["U"].values
K = energies["K"].values
dE = energies["dE"].values

# Load spatial grid
x = np.load(f"{folder_name}/x_grid.npy")

#--------------------------
# Harmonic oscillator ground state (target)
target = (1 / np.pi)**0.25 * np.exp(-x**2 / 2)
#---------------------------

n_frames = len(steps)

if animation == True:
    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    ax1, ax2, ax3 = ax

    line_psi_norm, = ax1.plot(x, psi_norm[0], label=r"$\psi_{\rm NQS}$")
    line_target, = ax1.plot(
        x, target, "--", label=r"$\psi_{\rm target}(\lambda=0)$"
    )
    ax1.set_title("Normalized wave function")
    ax1.legend()

    line_psi, = ax2.plot(x, psi_nn[0], label=r"$\psi_{\rm NQS}$")
    ax2.set_title("Raw NQS wave function")
    ax2.set_ylim(-0.1, 2.5)
    ax2.legend()

    line_E, = ax3.plot([], [], label="$E$")
    line_U, = ax3.plot([], [], label="$U$")
    line_K, = ax3.plot([], [], label="$K$")

    E0, n, dE0 = pertTheory_Energy(L) 
    E0_true = numericalEnergy_Schrodinger(L)

    ax3.set_xlim(0, steps[-1])
    ax3.set_ylim(ax_energy_lims[0], ax_energy_lims[1])
    # ax3.set_ylim(0.7,0.9)
    ax3.set_title("Energy convergence")

    #E0 - perturbation theory prediction
    #label with error
    label_text = rf"$E_0^{{({n})}} = {E0:.3f} \pm {dE0:.3f}$" 

    #plot horizontal line
    if L!=0:
        ax3.axhline(E0, color="black", label=label_text)
        ax3.axhline(E0_true, color='green', label=rf'$E_0^{{\rm num}}={E0_true:.4f}$', linestyle='dashed')
    else:
        ax3.axhline(E0, color="black", label=rf"$E_0={E0:.1f}$")
    
    #shaded error region
    ax3.axhspan(E0-dE0, E0+dE0,color='black',alpha=0.2)

    ax3.legend()

    #step label
    step_text = fig.text(0.5, 0.95, "Step: 0", ha="center", fontsize=12, color="red")
    E_band = None

    def update(frame):
        """
        Animation update function
        """
        global E_band

        # --- wavefunctions
        line_psi_norm.set_ydata(psi_norm[frame])
        line_psi.set_ydata(psi_nn[frame])

        # --- energies
        t = steps[:frame+1]
        line_E.set_data(t, E[:frame+1])
        line_U.set_data(t, U[:frame+1])
        line_K.set_data(t, K[:frame+1])

        # --- error band
        if E_band is not None:
            E_band.remove()

        E_band = ax3.fill_between(
            t,
            E[:frame+1] - dE[:frame+1],
            E[:frame+1] + dE[:frame+1],
            color=line_E.get_color(),
            alpha=0.4
        )

        # --- step counter
        step_text.set_text(f"Step: {steps[frame]}")

        return line_psi_norm, line_psi, line_E, line_U, line_K

    #create animation
    print("-- Creating animation...")
    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=50,
        blit=False
    )

    ani.save(f"{folder_name}/training_evolution.mp4", fps=fps,dpi=150)
    plt.close(fig)  # <-- closes the animation figure
    print("-- Animation saved.")
#-------------------------

if plot_last == True:
    #-------------------------
    # Plot last step of training
    #-------------------------
    print("\nWe plot now the last step of the training process:")
    final_frame = n_frames - 1  # index of the last step

    fig2, ax2 = plt.subplots(1, 3, figsize=(10, 4))
    ax1f, ax2f, ax3f = ax2

    # --- Normalized wavefunction + target
    ax1f.plot(x, psi_norm[final_frame], label=r"$\psi_{\rm NQS}$")
    ax1f.plot(x, target, "--", label=r"$\psi_{\rm target}(\lambda=0)$")
    ax1f.set_title("Normalized wave function")
    ax1f.legend()

    # --- Raw NQS wavefunction
    ax2f.plot(x, psi_nn[final_frame], label=r"$\psi_{\rm NQS}$")
    ax2f.set_title("Raw NQS wave function")
    ax2f.legend()

    # --- Energies with error band
    t = steps
    E_final = E
    dE_final = dE

    ax3f.plot(t, E_final, color='blue',label="$E$")
    # ax3f.plot(t, U, label="$U$")
    # ax3f.plot(t, K, label="$K$")

    # Error band
    ax3f.fill_between(t, E_final - dE_final, E_final + dE_final, color='blue', alpha=0.4)

    E0, n, dE0 = pertTheory_Energy(L)
    E0_true = numericalEnergy_Schrodinger(L)

    ax3f.set_xlim(0, steps[-1])
    # ax3f.set_ylim(0, 1)
    ax3f.set_ylim(ax_energy_lims[0], ax_energy_lims[1])
    ax3f.set_title("Energy convergence")

    #E0 - perturbation theory prediction
    #label with error
    label_text = rf"$E_0^{{({n})}} = {E0:.3f} \pm {dE0:.3f}$" 

    #plot horizontal line
    if L!=0:
        ax3f.axhline(E0, color="black", label=label_text)
        ax3f.axhline(E0_true, color='green', label=rf'$E_0^{{\rm num}}={E0_true:.4f}$')
    else:
        ax3f.axhline(E0, color="black", label=rf"$E_0={E0:.1f}$")

    #shaded error region
    ax3f.axhspan(E0-dE0, E0+dE0,color='black',alpha=0.2)

    # ax3f.axhline(0.551139, color='red')
    # ax3f.axhspan(0.551139-0.00113,0.551139+0.00113,color='red',alpha=0.2)

    # Step text
    ax3f.text(0.5, 0.95, f"Step: {steps[-1]}", transform=ax3f.transAxes,
            ha="center", fontsize=12, color="red")

    ax3f.legend(loc='lower right')
    plt.tight_layout()

    # --- Save figure as PNG ---
    fig2.savefig(f"{folder_name}/final_step.png", dpi=150)
    print(f"-- Final step figure saved in {folder_name}/final_step.png")

    plt.show() if show_plot else None

