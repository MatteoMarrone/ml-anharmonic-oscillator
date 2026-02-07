import numpy as np
import matplotlib.pyplot as plt

n, absA, p10 = np.loadtxt("data_pt.txt",unpack=True)
n = np.array(n, dtype=int)

A = (-1)**(n-1) * absA * 10**p10 #values of An

def pertTheory_Energy(lambda_val):
    """
    Compute the perturbative ground-state energy (scalar input).

    E0 = sum_{n=0}^{n_val} A_n * lambda^n
    with A_0 = 0.5

    n_val is chosen using the minimal lambda_max such that lambda < lambda_max.

    Returns:
        E0      : energy
        n_val   : maximal n used in the sum
        err_est : estimated truncation error (first omitted term)
    """

    # prepend A_0 = 0.5 to coefficients
    A_full = np.concatenate(([0.5], A))  # A[0] -> A_1 in old notation

    # n = 0, 1, ..., len(A)
    n = np.arange(len(A_full), dtype=int)

    # lambda_max via ratio test for n >= 0
    # lmax[n] = |A_n / A_{n+1}|, so length = len(A_full)-1
    lmax = np.abs(A_full[:-1] / A_full[1:])

    # Find truncation order
    mask = lambda_val < lmax
    if not np.any(mask):
        # raise ValueError(f"No lambda_max satisfies lambda = {lambda_val}")
        return A_full[0], 0, A_full[1]*lambda_val

    idx_candidates = np.where(mask)[0]
    idx = idx_candidates[np.argmin(lmax[idx_candidates])]
    n_val = n[idx]

    # Compute energy sum
    powers = lambda_val ** np.arange(n_val + 1)  # 0..n_val
    series_sum = np.sum(A_full[:n_val + 1] * powers)

    E0 = series_sum

    # --- Error estimate: 0.5 * first omitted term ---
    if n_val + 1 < len(A_full):
        err_est = np.abs(A_full[n_val + 1] * lambda_val ** (n_val + 1))
    else:
        err_est = np.nan  # no higher-order coefficient available

    return E0, n_val, err_est


# l_array = np.linspace(0,0.1,200)
# E = np.array([pertTheory_Energy(l)[0] for l in l_array])
# dE = np.array([pertTheory_Energy(l)[2] for l in l_array])
# plt.errorbar(l_array,E,dE,alpha=0.5)
# plt.show()