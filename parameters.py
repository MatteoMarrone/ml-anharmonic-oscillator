#-----------------------------------------
import torch

# Network hyperparameters
Nin = 1   # Inputs to the neural network
Nout = 1  # Outputs of the neural network
Nhid = 4  # Nodes in the hidden layer (only 1 hidden layer)

# Training hyperparameters
epochs = 100
lr = 1 # Learning rate

# Mesh parameters (animation purposes)
Nx = 120                    # Mesh division
train_a = -8                      # Mesh lower limit
train_b = 8                        # Mesh upper limit

# Network parameters.
seed = 5 # Seed of the random number generator
torch.manual_seed(seed)
scale = 1
W1 = scale * (2 * torch.rand(Nhid, Nin, requires_grad=True) + 1) # First set of coefficients
    #This creates a tensor of shape (Nhid=4, Nin=1) = (4,1)
    #The function torch.rand generates random numbers between [0,1)
    #The condition requires_grad=True tells PyTorch "I want to compute gradients for this tensor during backpropagation."

B = scale * (2 * torch.rand(Nhid, requires_grad=True) - 1)   # Set of bias parameters
    # The biases are random numbers between [-1,1), with shape (4,)
W2 = (torch.rand(Nout, Nhid, requires_grad=True))       # Second set of coefficients
    #This tensor now has shape (Nout=1,Nhid=4) = (1,4).

#Markov chains parameters
n_samples = 1000
sigma = 1
skip_size = 5
burn_in = 500
SAVE_CHAIN_EVERY = 1 #we save in a file the mc chain values at every epoch

#Physical model parameters:
L = 0.1 #value of lambda, anharmonic oscillator
