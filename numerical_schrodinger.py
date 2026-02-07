import numpy as np
from scipy.linalg import eigh_tridiagonal
import matplotlib.pyplot as plt

from parameters import L
from perturbed_energy import pertTheory_Energy

single = False #if single, it calculates E(lambda) for a single value of lambda
loop = False #if loop, it calculates E(lambda) for different values of lambda

def numericalEnergy_Schrodinger(lam, L=10.0, N=2000):
    """
    Compute the ground-state energy of the 1D anharmonic oscillator
    V(x) = 0.5 x^2 + lam x^4
    using finite differences.

    Parameters
    ----------
    lam : float
        Anharmonic coupling lambda
    L : float, optional
        Half-width of the spatial domain [-L, L]
    N : int, optional
        Number of grid points

    Returns
    -------
    E0 : float
        Ground-state energy
    """

    # spatial grid
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]

    # potential
    V = 0.5 * x**2 + lam * x**4

    # Hamiltonian (tridiagonal)
    diag = 1.0 / dx**2 + V
    offdiag = -0.5 / dx**2 * np.ones(N - 1)

    # solve eigenvalue problem
    E, _ = eigh_tridiagonal(diag, offdiag)

    return E[0]

if single:
    lambda_value = L
    E0 = numericalEnergy_Schrodinger(lambda_value)
    print(f"lambda = {lambda_value} - gs energy: {E0:.4f}")

# --------------------
if loop:
    # loop
    # parameters
    lambda_array = np.linspace(0,0.66,30)
    E_array = []
    E_array_pt = []
    dE_array_pt = []
    L = 10.0
    N = 2000

    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]

    step = 0
    for lam in lambda_array:

        step+=1
        # potential
        V = 0.5*x**2 + lam*x**4

        # build tridiagonal Hamiltonian
        diag = 1.0/dx**2 + V
        offdiag = -0.5/dx**2 * np.ones(N-1)

        # solve eigenvalue problem
        E, psi = eigh_tridiagonal(diag, offdiag)

        print(f"step = {step}")
        # print(f"lambda = {lam}, Ground state energy: {E[0]:.4f}")
        E_array.append(E[0])
        E_array_pt.append(pertTheory_Energy(lam)[0])
        dE_array_pt.append(pertTheory_Energy(lam)[2])

    E_array = np.array(E_array)
    E_array_pt = np.array(E_array_pt)
    dE_array_pt = np.array(dE_array_pt)

    plt.tight_layout()
    plt.plot(lambda_array, E_array, label=r'numerical')
    plt.errorbar(lambda_array,E_array_pt, dE_array_pt, fmt='o',label=r'perturbation')
    plt.legend(loc='lower left')
    plt.xlabel(r'$\lambda$',fontsize='large')
    plt.ylabel(r'$E(\lambda)$',fontsize='large')
    # plt.savefig("figures\energy_numerical_pt_lambda.png",dpi=300)
    plt.show()
