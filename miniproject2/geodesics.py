import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import torch


M = 7
K = 2
RESOLUTION = 10

def g(t, W, K):
    w_iK = 0
    total = 0
    for k in range(1, K):
        w_ik = W[k]
        w_iK -= w_ik
        
        total += w_ik * t ** k + 0
        
    total += w_iK * t ** K + 0
    
    return total
    
def get_curve(t, point1, point2, W = np.random.randn(K) * M) -> torch.tensor:
    return torch.tensor((1-t)*point1 +t*point2) #Line
    # return (1-t)*point1 +t*point2 + g(t, W, K) #Polynomial

def energy_curve(curve): #Monte Carlo Energy
    n_samples = curve.shape[0] 
    return torch.sum(torch.stack([torch.norm(curve  [idx+1] - curve   [idx]) ** 2 for idx in range(n_samples - 1)])) / n_samples
