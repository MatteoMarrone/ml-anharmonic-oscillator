
#MODEL
#-------------------
import torch
from torch import nn
import math
from parameters import *

class HarmonicNQS(nn.Module):
    def __init__(self, W1, B, W2):
        super(HarmonicNQS, self).__init__() #This initializes the parent nn.Module
        
        # We set the operators 
        self.lc1 = nn.Linear(in_features=Nin, 
                             out_features=Nhid, 
                             bias=True)   # shape = (Nhid, Nin)

            #This first linear layer transforms the input x to z1 = W1.x + B with W1 the first weights and B the biases
            #since bias=True.
        
        self.actfun = nn.Sigmoid()        # activation function. 
            #This is a sigmoid, a1 = sigma(z1) = 1/(1+exp(-z1))
        
        self.lc2 = nn.Linear(in_features=Nhid, 
                             out_features=Nout, 
                             bias=False)  # shape = (Nout, Nhid)

            #This second linear layer transforms the activation function a1 to the output, without biases,
            #this is o = W2.a1
        
        # We set the parameters 
        with torch.no_grad():
            self.lc1.weight = nn.Parameter(W1)
            self.lc1.bias = nn.Parameter(B)
            self.lc2.weight = nn.Parameter(W2)

    # We set the architecture
    def forward(self, x): 
        o = self.lc2(self.actfun(self.lc1(x)))
            #This means, mathematically, O(x) = W2.sigmoid(W1.x + B)

        return o * torch.exp(-0.5*x.pow(2)) #gaussian envelope

class HarmonicExact(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = math.pi ** (-0.25)

    def forward(self, x):
        return self.norm * torch.exp(-0.5 * x.pow(2))