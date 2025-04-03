import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import torch
import torch.nn as nn
from tqdm import tqdm
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

# def energy_curve_monte_carlo(path_z, decoder):
#     x = decoder(path_z).mean
#     total_energy = 0.0
#     for i in range(x.shape[0] - 1):
#         diff = x[i+1] - x[i]
#         total_energy += torch.sum(diff**2)
#     return total_energy / (x.shape[0]-1)

def energy_curve_monte_carlo(path_z, decoder):
    path_z = path_z.view(-1, 2)  # Reshape to [n_samples, 2]
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

def ensemble_curve_energy_monte_carlo(path_z, decoders, n_samples=10):
    total_energy = torch.zeros((), device=path_z.device, dtype=path_z.dtype, requires_grad=True)

    for i in range(path_z.shape[0] - 1):
        z_i   = path_z[i].unsqueeze(0)     # shape (1, latent_dim)
        z_ip1 = path_z[i+1].unsqueeze(0)   # shape (1, latent_dim)
        segment_energy = torch.zeros((), device=path_z.device, dtype=path_z.dtype, requires_grad=True)
        for _ in range(n_samples):
            l = random.randint(0, len(decoders) - 1)
            k = random.randint(0, len(decoders) - 1)
            dist_l = decoders[l](z_i)     
            dist_k = decoders[k](z_ip1)
            diff = dist_l.mean - dist_k.mean 
            sq_norm = (diff**2).sum()
            segment_energy = segment_energy + sq_norm
        segment_energy = segment_energy / n_samples
        total_energy = total_energy + segment_energy
    return total_energy

def find_ensemble_geodesic(
    z_start, z_end, decoders,
    steps=1000, lr=0.1, path_points=20, n_samples=10
):
    """
    Find a geodesic between z_start and z_end using ensemble-based curve energy
    (pull-back metric approx from multiple decoders).
    """
    device = z_start.device

    t_vals = torch.linspace(0, 1, path_points, device=device)
    init_path = [(1 - t) * z_start + t * z_end for t in t_vals]
    path_z = nn.Parameter(torch.stack(init_path), requires_grad=True)

    optimizer = torch.optim.Adam([path_z], lr=lr)

    with tqdm(range(steps), desc="Geodesic optimization") as pbar:
        for step in pbar:
            optimizer.zero_grad()

            # Clamp endpoints
            with torch.no_grad():
                path_z[0] = z_start
                path_z[-1] = z_end

            # Compute energy
            energy_tensor = ensemble_curve_energy_monte_carlo(path_z, decoders, n_samples=n_samples)
            # Backprop
            energy_tensor.backward()
            optimizer.step()

            pbar.set_postfix({"energy": float(energy_tensor.item())})

    with torch.no_grad():
        path_z[0] = z_start
        path_z[-1] = z_end

    final_energy = ensemble_curve_energy_monte_carlo(path_z, decoders, n_samples=n_samples)
    return path_z.detach(), final_energy.item()

def euclidean_path_length(path_z):
    """
    Computes the Euclidean length of a discretized path in latent space.
    That is, sum of ||z_{i+1} - z_i|| over i.

    Args:
        path_z (torch.Tensor): shape (T, latent_dim), the optimized path.

    Returns:
        float: the total path length in latent space (Euclidean).
    """
    total_length = 0.0
    # We iterate over each adjacent pair in path_z
    for i in range(path_z.shape[0] - 1):
        segment_len = torch.norm(path_z[i+1] - path_z[i]).item()
        total_length += segment_len
    return total_length

def coefficient_of_variation(distances):
    """
    distances: list of floats, e.g. [d_ij^(1), ..., d_ij^(M)] from M seeds
    Returns the CoV = std / mean
    """
    import statistics
    mean_ = statistics.mean(distances)
    std_  = statistics.stdev(distances)
    return std_ / mean_ if mean_ > 1e-10 else 0.0
