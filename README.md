<!-- # ml-anharmonic-oscillator
  -->

# *Machine learning the anharmonic oscillator*
<img width="1790" height="703" alt="image" src="https://github.com/user-attachments/assets/35bde994-0f96-4424-b825-e65d2c6c0834" />
(Picture taken from Javier Rozalen's github).

## General information
This repository contains several Python files for computing the ground state of the one-dimensional quantum anharmonic oscillator using variational Monte Carlo with a neural-network wavefunction. The implementation explores machine learning based variational ansatz to approximate the ground state energy of non-analytical quantum theories.

The theory we are dealing with is that of,

$\hat H = -\frac{1}{2}\frac{d^2}{dx^2} + V(x)$ 

with a potential given by,

$` V(x) = \frac{1}{2} x^2 + \lambda x^4 `$

in natural units.

Repository structure:
```
.
├── animation.py                              # Animation and plots of the simulation results
├── data_pt.txt                               # Coefficients of perturbation series for E(lambda) [Bender & Wu - 1969]
├── env.yml                                   # Conda environment
├── mc_chain.py                               # Construct Markov Chain
├── LICENSE                                   # License
├── README.md
├── model.py                                  # Neural Network model
├── parameters.py                             # Relevant parameters of the simulation
├── perturbed_energy.py                       # Calculate energy in perturbation theory
├── numerical_schrodinger.py                  # Calculate E(lambda) numerically
├── train.py                                  # MAIN FILE. Training of the neural network.
└── requirements.txt                          # requirements
```

## Installation 
To install this project in your computer, follow these steps:
1. Download the folder of this repository directly into your own computer.
2. Create a conda enviroment from the `.yml` file in the repository:

`conda env create -f env.yml`

4. Activate the environment:

`conda activate ml-osc`

5. Install further requirements:

`pip install -r requirements.txt`

## Usage
The main files of this folder consist on:
- **`train.py`**. This is the main script for training the neural quantum wavefunction. It performs the full training loop for the neural network, optimizing the parameters in the direction of minimum gradient, to minimize the energy, and outputs the computed energy at each epoch, until reaching the maximum number of epochs specified. The way the minimization algorithm works is by following the direction of minimum gradient in parameter space. Let $\{\alpha\}$ be the parameter set, then,

  $`\alpha_{\rm new} = \alpha_{\rm old} - \eta \nabla_\alpha E`$ (1)
  
  with $\alpha_{\rm new}$ being the new set of parameters after an epoch, $\nabla_\alpha E$ the gradient of the loss function (energy) in parameter space, and $\eta$ what is called the **learning rate**, which is set to 1 by default.

  When running this file, at the beginning the terminal will output the energy estimation in perturbation theory for that particular value of $\lambda$, and also the numerical estimation, calculated using the functions defined in the file `numerical_schrodinger.py`. Then, the training process begins, where equation (1) is applied iteratively until reaching the last epoch of simulation, or eventually until the user presses `Ctrl+C` to stop the simulation. In this case, the data will be saved safely in the corresponding files. The data of a simulation with certain $\lambda$ will be saved in a folder as follows:

```
├──results
  ├── (lambda=0.000)
    ├── energies.csv        # File where: step, kinetic energy, potential energy, total energy, monte carlo error, are stored in different columns.
    ├── metadata.json       # File where all the relevant parameters of the simulation are stored.
    ├── psi_nn.npy          # Values of the raw (unnormalized) wavefunction at each epoch are stored, for later animation and plot purposes.
    ├── psi_morm.npy        # Values of the normalized wavefunction at each epoch are stored, for later animation and plot purposes.
    ├── x_grid.npy          # Values of the grid. For animation and plot purposes.
```

- **`animation.py`**. This file is intended to be used after a simulation is already run. In this file there are a couple of important flags to mention:
  - `animation = True`. If set to `True`, it will create a `.mp4` file of the evolution, where the user will be able to see the wavefunction convergence, and the energy convergence towards the expected value.
  - `plot_last = True`. If set to `True`, a `.png` file will be created that plots the evolution of the wavefunction and energy convergence in a single figure.
  - `show_plot = False`. If set to `True`, the line `plt.show()` will be called and a pop-up of the animation video and plot will appear. For simplicity, we set it to `False` by default.
  
  There is also a variable called `ax_energy_lims` that sets the y-axis limit for the energy subfigure. Also, a `fps` variable is defined at the beginning to play with the velocity at which the video saved is being displayed.

- **`mc_chain.py`**. In this file, the function `mh_samples(...)` is defined, which creates a Markov chain of samples following the distribution $`|\psi(x)|^2`$ at each epoch. The relevant Markov chain parameters, such as,
  - Number of samples of the chain `n_samples`. Corresponds to the total number of samples of the chain.
  - Burn-in `burnin`,
  - Number of samples we skip, `skip_size`. We do this to reduce autocorrelation of the chain.
  - The value of $\sigma$ of the proposal distribution, `sigma`. The proposal distribution is a Gaussian distribution centered at the previous value, and with standard deviation given by $\sigma$.
  
  These are all inputs of this function, and they can be tuned to whatever value wanted directly in the file `parameters.py`.

  In the same file, there is a flag called `run = False` that is set to `False` by default. When set to `True`, the program will run some diagnostics of the Markov chain for the first epoch of the training process (i.e. the initial value of the wavefunction), and the user will see some plots of samples vs index, or an histogram of density vs sample value, which can be compared to the desired $`|\psi(x)|^2`$.

- **`model.py`**. In this file, the Neural Network is defined. It consists of a single input `Nin=1` (value of $x$) and single output `Nout=2` (value of $\psi(x)$), with a hidden layer of four nodes. We have in total twelve parameters: `Nhid=4` biases $B_k$, `Nhid=4` weights between the input and the hidden layer, $`(W_1)_k`$, and `Nhid=4` weights between the hidden layer and the output, $`(W_2)_k`$. The activation function is a sigmoid, $\sigma(x) = (1+e^{-x})^{-1}$. The output can be written in terms of these parameters as:

$` \psi_{\rm NN}(x) = \sum_{k=1}^{N_{\rm hid}} W_{2,k}\sigma({(W_1)}_k x + B_{k}) e^{-x^2/2} `$

where the final result is being multiplied by a Gaussian envelope $e^{-x^2/2}$. In the same model file, a class called `HarmonicExact` is defined, which outputs the true ground state of the harmonic ($\lambda=0$) oscillator. It was added for debugging purposes, but it remained ever since.

- **`numerical_schrodinger.py`**. This file computes the ground-state energy of the 1D anharmonic oscillator using finite difference discretization of the Schrodinger equation and solving the resulting tridiagonal eigenvalue problem. The function `numericalEnergy_Schrodinger(lam, L, N)` returns the lowest eigenvalue, corresponding to the ground state. The grid is 1D from $`[-L,L]`$ with $L=10$ by default, and $N=2000$ number of points by default. These values can be changed, though the ground state energy will not change considerably.

  There are two flags.
  - `single = False`. If set to `True`, the program will output the value of $\lambda$ and the corresponding ground state energy.
  - `loop = False`. If set to `True`, it will calculate the ground state energy for a wide range of values of $\lambda$, and plot the corresponding curve $E(\lambda)$.

- **`perturbed_energy.py`**. This file calculates the energy using perturbation theory. In this formalism, the energy can be written as a sum of powers of the coupling $\lambda$ as

$` E(\lambda) = \sum_{n=0}^\infty A_n \lambda^n `$

  where $A_0=0.5$ is the usual harmonic oscillator energy. What this program does initially is read the file `data_pt.txt` to construct the coefficients $A_n$. This file is composed of three columns:
  - First column: Order $n$ of the series.
  - Second column: Absolute value of $A_n$ up to some power of 10.
  - Third column: The corresponding power of 10 of the coefficient.

  The sign of the coefficients is alternating, $(-1)^{n-1}$ for any $n>1$. These table is constructed from the results of [Bender & Wu (1969)](https://journals.aps.org/pr/abstract/10.1103/PhysRev.184.1231).

  After having constructed the coefficients $A_n$, the perturbation series estimation for $E(\lambda)$ is obtained from the sum up to $n_{\rm max}$, which is the maximal order at which perturbation series is meaningful for a given value of $\lambda$. Since this series is divergent for any $\lambda$ (it is what it is), the method we employ to estimate this value of energy is sum up until $n_{\rm max}$, which is obtained from the maximum value of the coupling such that the next-to-leading-order term in the series is not greater than the previous one, i.e $\lambda_{\rm max} = |A_{n}/A_{n+1}|$, and $n_{\rm max}$ is the maximal order of perturbation series that satifies the constraint $\lambda<\lambda_{\rm max}$.

- **`parameters.py`**. In this file, the user will find all the relevant parameters of the simulation, such as:
  - Network hyperparameters:
    - `Nin=1`. Inputs to the neural network. Set to 1 by default, and we encourage the user **not to change** this value.
    - `Nout=1`. Outputs of the neural network. Set to 1 by default, and we encourage the user **not to change** this value.
    - `Nhid=4`. Nodes of the hidden layer. It can be changed to other values.
  - Training hyperparameters:
    - `epochs=100`. Maximum number of epochs per simulation. For slightly larger values of $\lambda$, we recommend the user to increase this value. Also, if the simulation seems to have converged way sooner than expected, the user can press `Ctrl+C` safely, and all the data of the simulation up until the stopping epoch will be saved in their corresponding files.
    - `lr=1`. Learning rate. After several tests, we recommend the user to work with values close to 1.
  - Mesh parameters: **This is only for animation purposes**.
    - `Nx=120`. Number of points of $\psi(x)$ in the grid.
    - `train_a=-8`. Leftmost point of the grid.
    - `train_b=+8`. Rightmost point of the grid.
  - Network parameters:
    - `seed=5`. Seed of the random number generator. It can be changed to whatever value the user may want to try.
    - `W1`. The first set of weights of the neural network. Initialized randomly between [1,3].
    - `B`. The set of biases of the neural network. Initialized randomly between [-1,1].
    - `W2`. Second set of weights of the neural network. Initialized randomly between [0,1).
  - Markov chain parameters:
    - `n_samples=1000`. Total number of samples for the Markov chain (after burn in and thinning).
    - `sigma=1`. Value of standard deviation of proposal distribution.
    - `skip_size=5`. Number of samples we skip (thinning).
    - `burn_in=500`. We burn the first `burn_in` samples generated, to reduce correlation with the initial value of the chain.
    - `save_chain = True`. When set to true, the values of the Markov chain at each epoch of the training process will be saved in a file called `chains.h5`. This file, as one may expect, can be quite large, so if the user does not want to have this file created in the first place, set this flag to `False`. Otherwise, set to `True`.
    - `SAVE_CHAIN_EVERY=1`. At how many epochs of the training process we choose to save the values of the Markov chain. Set to 1 (i.e. at every epoch) by default.
  - Physical model features:
    - `L=0.`. Value of the coupling $\lambda$ for the anharmonic oscillator.


## Uninstall
To remove the virtual environment created follow the steps below:
1. Make sure your current environment is not `ml-osc`, or if it is, type:

`conda deactivate`

2. Remove the environment.

`conda remove -n ml-osc --all`

3. Optional: Delete the folder on your computer.

## Support 
If you have any questions or issues, please contact at matteomarrone27@gmail.com
