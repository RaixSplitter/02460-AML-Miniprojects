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

def energy_curve_monte_carlo(path_z, decoder):
    x = decoder(path_z).mean
    total_energy = 0.0
    for i in range(x.shape[0] - 1):
        diff = x[i+1] - x[i]
        total_energy += torch.sum(diff**2)
    return total_energy / (x.shape[0]-1)

def energy_curve_with_metric(curve_z, decoder):
    total_energy = torch.tensor(0., device=curve_z.device, dtype=curve_z.dtype)

    for i in range(curve_z.shape[0] - 1):
        z_i = curve_z[i].unsqueeze(0)  # shape [1,2]
        z_next = curve_z[i+1].unsqueeze(0)
        delta = (z_next - z_i).squeeze(0)
        
        x_i = decoder(z_i).mean.view(1, -1)

        # Build the Jacobian of x_i wrt z_i
        J_rows = []
        for out_idx in range(x_i.shape[1]):
            grad_out = torch.autograd.grad(
                x_i[0, out_idx],
                z_i,
                retain_graph=True
            )[0]  # shape [1,2]
            J_rows.append(grad_out)
        J_f = torch.cat(J_rows, dim=0)  # shape [784,2]

        g_i = J_f.t() @ J_f  # shape [2,2]

        seg_energy = delta @ g_i @ delta
        total_energy += seg_energy

    return total_energy