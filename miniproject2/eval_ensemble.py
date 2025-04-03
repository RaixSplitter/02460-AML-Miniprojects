import hydra
from models import EnsembleVAE, GaussianPrior, GaussianDecoder, GaussianEncoder, train
import os
import random
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.nn as nn
from tqdm import tqdm
from omegaconf import DictConfig
from matplotlib import pyplot as plt


def coefficient_of_variation(distances):
    """
    distances: list of floats, e.g. [d_ij^(1), ..., d_ij^(M)] from M seeds
    Returns the CoV = std / mean
    """
    import statistics
    mean_ = statistics.mean(distances)
    std_  = statistics.stdev(distances)
    return std_ / mean_ if mean_ > 1e-10 else 0.0


def g(t, W):
    w_iK = 0
    total = 0
    K = len(W)
    for k in range(1, K):
        w_ik = W[k]
        w_iK -= w_ik

        total += w_ik * t**k + 0

    total += w_iK * t**K + 0

    return total


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

def get_curve(t, point1, point2, W) -> torch.tensor:
    return (1 - t) * point1 + t * point2 + g(t, W)  # Line


def energy_curve_monte_carlo(curve_z, decoder, num_t=30):

    curve_z = curve_z.view(-1, 2)  # Reshape to [n_samples, 2]
    x = decoder(curve_z).mean
    total_energy = 0.0
    for i in range(x.shape[0] - 1):
        diff = x[i + 1] - x[i]
        total_energy += torch.sum(diff**2)
    return total_energy / (x.shape[0] - 1)

def ensemble_curve_energy_monte_carlo(curve_z, decoders, num_t=30, n_samples=5):
    total_energy = torch.zeros((), device=curve_z.device, dtype=curve_z.dtype, requires_grad=True)
    for i in range(curve_z.shape[0] - 1):
        z_i   = curve_z[i].unsqueeze(0)     # shape (1, latent_dim)
        z_ip1 = curve_z[i+1].unsqueeze(0)   # shape (1, latent_dim)

        # We'll accumulate this segment's energy in a local tensor
        segment_energy = torch.zeros((), device=curve_z.device, dtype=curve_z.dtype, requires_grad=True)

        for _ in range(n_samples):
            l = random.randint(0, len(decoders) - 1)
            k = random.randint(0, len(decoders) - 1)

            dist_l = decoders[l](z_i)     # distribution => has .mean
            dist_k = decoders[k](z_ip1)

            diff = dist_l.mean - dist_k.mean  # shape (1, 1, H, W) for images
            sq_norm = (diff**2).sum()
            segment_energy = segment_energy + sq_norm

        # average over the n_samples
        segment_energy = segment_energy / n_samples
        total_energy = total_energy + segment_energy

    return total_energy
    pass


def find_ensemble_geodesic(z_start, z_end, decoders, cfg):
    W = torch.nn.Parameter(torch.rand(cfg.geodesics.k) * cfg.geodesics.magnitude, requires_grad=True)
    optimizer = torch.optim.Adam([W], lr=cfg.geodesics.lr)
    
    for step_i in range(cfg.geodesics.steps):
        optimizer.zero_grad()

        # Get the curve
        curve_z = torch.stack([get_curve(t, z_start, z_end, W) for t in np.linspace(0, 1, cfg.geodesics.num_t)])

        # Compute the energy of the curve
        energy = ensemble_curve_energy_monte_carlo(curve_z, decoders, num_t=cfg.geodesics.num_t, n_samples=cfg.n_samples)

        # Backpropagate
        energy.backward()
        optimizer.step()
    
    
    final_energy = ensemble_curve_energy_monte_carlo(curve_z, decoders, num_t=cfg.geodesics.num_t)
    return curve_z, final_energy

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig, SEED=42) -> None:
    device = cfg.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

    # Set seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=cfg.batch_size, shuffle=False
    )

    # Define prior distribution
    M = cfg.latent_dim

    def new_encoder():
        encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.Softmax(M),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(M),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )
        return encoder_net

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softmax(M),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softmax(M),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softmax(M),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net

    data_iter = iter(mnist_test_loader)
    x_full, y_full = next(data_iter)  # just one batch for demonstration
    # pick 10 random pairs
    pairs_idx = random.sample(
        range(len(x_full)), 20
    )  # 20 so we form 10 pairs
    pairs_idx = [
        (pairs_idx[i], pairs_idx[i + 1])
        for i in range(0, 20, 2)
    ]  # Step 2

    # We'll store the final average CoVs for plotting
    eucl_covs_for_plot = []
    geo_covs_for_plot = []

    dec_list = [1, 2, 3]
    for num_decs in dec_list:
        print("Initializing model with", num_decs, "decoders")
        all_eucl = []
        all_geo = []

        model_files = os.listdir(f"{cfg.experiment_folder}/ND{num_decs}")
        model_paths = [
            f"{cfg.experiment_folder}/ND{num_decs}/" + file
            for file in model_files
            if file.endswith(".pt")
        ]
        pbar = tqdm(model_paths, total=len(model_paths), desc="Loading models")
        for model_path in pbar:
            # Load the model
            model = EnsembleVAE(
                GaussianPrior(M),
                GaussianDecoder(new_decoder()),
                GaussianEncoder(new_encoder()),
                num_decs,
            ).to(cfg.device)
            model.load_state_dict(
                torch.load(model_path, weights_only=True, map_location=cfg.device),
                strict=False,
            )
            model.eval()

            # Get the decoders
            decoders = model.decoders

            # Freeze the model parameters
            for decoder in decoders:
                decoder.requires_grad = False
            model.encoder.requires_grad = False

            # 1) Get the latent points for the pairs
            # 2) Compute the distances

            # Distances for each pair => lists
            eucl_dists_for_seed = []
            geo_dists_for_seed = []

            for idx_a, idx_b in pairs_idx:
                ya = x_full[idx_a : idx_a + 1].to(cfg.device)
                yb = x_full[idx_b : idx_b + 1].to(cfg.device)

                with torch.no_grad():
                    za = model.encoder(ya).mean.squeeze(0)
                    zb = model.encoder(yb).mean.squeeze(0)
                    # shape (latent_dim,)

                # Euclidean distance
                eucl_dist = torch.norm(za - zb).item()
                
                path_opt, final_energy = find_ensemble_geodesic(z_start=za, z_end=zb, decoders=decoders, cfg=cfg)

                geo_dist = euclidean_path_length(path_opt)

                eucl_dists_for_seed.append(eucl_dist)
                geo_dists_for_seed.append(geo_dist)

            # accumulate
            all_eucl.append(eucl_dists_for_seed)
            all_geo.append(geo_dists_for_seed)

        # shape => (n_seeds, n_pairs) => transpose => (n_pairs, n_seeds)
        all_eucl = np.array(all_eucl).T
        all_geo  = np.array(all_geo).T

        # compute CoV per pair, then average
        pair_eucl_covs = [coefficient_of_variation(all_eucl[i]) for i in range(all_eucl.shape[0])]
        pair_geo_covs  = [coefficient_of_variation(all_geo[i]) for i in range(all_geo.shape[0])]

        avg_cov_eucl = np.mean(pair_eucl_covs)
        avg_cov_geo  = np.mean(pair_geo_covs)

        print(f"=== {num_decs} decoders ===")
        print(f"Avg CoV (Eucl) over {len(pairs_idx)} pairs: {avg_cov_eucl:.4f}")
        print(f"Avg CoV (Geo ) over {len(pairs_idx)} pairs: {avg_cov_geo:.4f}")

        eucl_covs_for_plot.append(avg_cov_eucl)
        geo_covs_for_plot.append(avg_cov_geo)

    # 3) Plot CoV vs. number of decoders
    plt.plot(dec_list, eucl_covs_for_plot, marker='o', label='Eucl CoV')
    plt.plot(dec_list, geo_covs_for_plot, marker='o', label='Geo CoV')
    plt.xlabel('Number of decoders in ensemble')
    plt.ylabel('Coefficient of Variation')
    plt.legend()
    plt.title('CoV of distances across seeds vs. ensemble size')
    plt.savefig('cov_vs_num_decoders.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
    