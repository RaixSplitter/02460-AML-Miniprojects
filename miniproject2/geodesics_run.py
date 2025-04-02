import hydra
from matplotlib.legend_handler import HandlerTuple
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
import seaborn as sns

sns.set_theme(style="darkgrid")


def g(t, W):
    w_iK = 0
    total = 0
    K = len(W)
    for k in range(1, K):
        w_ik = W[k]
        w_iK -= w_ik
        
        total += w_ik * t ** k + 0
        
    total += w_iK * t ** K + 0
    
    return total
    
def get_curve(t, point1, point2, W) -> torch.tensor:
    return (1-t)*point1 +t*point2 + g(t, W) #Line

def energy_curve_monte_carlo(curve_z, decoder, num_t=30):
    
    
    curve_z = curve_z.view(-1, 2)  # Reshape to [n_samples, 2]
    x = decoder(curve_z).mean
    total_energy = 0.0
    for i in range(x.shape[0] - 1):
        diff = x[i+1] - x[i]
        total_energy += torch.sum(diff**2)
    return total_energy / (x.shape[0]-1)


    

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig, SEED = 42) -> None:
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
    
    # Init model
    model = EnsembleVAE(
        GaussianPrior(M),
        GaussianDecoder(new_decoder()),
        GaussianEncoder(new_encoder()),
        cfg.num_decoders,
    ).to(cfg.device)
    model.load_state_dict(
        torch.load(os.path.join(cfg.experiment_folder, cfg.model_name), weights_only=True)
    )
    model.eval()  # Set layer behavior to eval mode, note that it doesn't disable gradient calculation

    # Freeze the model parameters
    for decoder in model.decoders:
        decoder.requires_grad = False
    model.encoder.requires_grad = False

    # Grab N_POINTS Latents
    all_latents = []
    all_labels = []
    with torch.no_grad():
        for x, y in mnist_test_loader:
            x = x.to(device)
            latent = model.get_latent(x).detach()  # Explicitly detach latent
            all_latents.append(latent.cpu())
            all_labels.append(y.cpu())

            if not cfg.geodesics.n_points:  # If N_POINTS is not set, we want to sample all points
                continue
            if len(all_latents) * cfg.batch_size >= cfg.geodesics.n_points:  # If we have enough points, break
                break

    # Concatenate all latents and labels
    latent = torch.cat(all_latents, dim=0)
    labels = torch.cat(all_labels, dim=0)
    if cfg.geodesics.n_points:  # Limit to N_POINTS
        latent = latent[:cfg.geodesics.n_points]
        labels = labels[:cfg.geodesics.n_points]

    # Get latent pairs
    n = latent.shape[0]

    chosen_idx_pairs = random.choices(range(n), k=cfg.geodesics.n_latent_pairs * 2)
    chosen_idx_pairs = [
        (chosen_idx_pairs[i], chosen_idx_pairs[i + 1])
        for i in range(0, len(chosen_idx_pairs), 2)
    ]
    
    chosen_pairs = [
        torch.stack([latent[i], latent[j]]) for (i, j) in chosen_idx_pairs
    ]
    chosen_pairs = torch.stack(chosen_pairs).to(
        device
    )  # shape [N_LATENT_PAIRS, 2, 2]

    geodesic_coords_saved = []
    pbar = tqdm(total=len(chosen_pairs) * cfg.geodesics.steps, desc="Geodesics", disable = cfg.tqdm_disable)

    for i, (z_start, z_end) in enumerate(chosen_pairs):
        W = torch.nn.Parameter(torch.rand(cfg.geodesics.k) * cfg.geodesics.magnitude, requires_grad=True)
        
        optimizer = torch.optim.Adam([W], lr=cfg.geodesics.lr)

        pbar.set_description(f"Geodesics: Starting optimization")
        for step_i in range(cfg.geodesics.steps):
            optimizer.zero_grad()
            
            # Sample a random decoder from the ensemble
            decoder = random.choice(model.decoders)
            # Get the curve
            curve_z = torch.stack([get_curve(t, z_start, z_end, W) for t in np.linspace(0, 1, cfg.geodesics.num_t)])
            E = energy_curve_monte_carlo(curve_z, decoder, num_t = cfg.geodesics.num_t)
            E.backward()

            optimizer.step()
            pbar.set_description(f"Geodesics: {step_i+1}/{cfg.geodesics.steps} | Latent pair {i+1}/{cfg.geodesics.n_latent_pairs} | Energy: {E.item():.4f}")
            pbar.update(1)

        curve_z = torch.stack([get_curve(t, z_start.detach(), z_end.detach(), W) for t in np.linspace(0, 1, cfg.geodesics.num_t)])
        geodesic_coords = curve_z.detach().cpu().numpy()
        geodesic_coords_saved.append(geodesic_coords)
        
    
    geodesic_coords_saved = np.array(geodesic_coords_saved)
    np.save("latents.npy", latent.cpu().numpy())
    np.save("geodesics.npy", geodesic_coords_saved)
    np.save("labels.npy", labels.cpu().numpy())
    
    for geodesic_coords in geodesic_coords_saved:
        pg_plot = plt.plot(geodesic_coords[:, 0], geodesic_coords[:, 1], "-b", label = 'Pullback Geodesic')
        sl_plot = plt.plot([geodesic_coords[-1, 0], geodesic_coords[-1, 0]], [geodesic_coords[-1, 1], geodesic_coords[-1, 1]], "--r", label = 'Straight Line')

    
    plt.scatter(latent[:, 0].cpu(), latent[:, 1].cpu(), c=labels, label=labels, cmap="viridis", alpha=0.5)
    filename = f"results/geoT{cfg.geodesics.num_t}LP{cfg.geodesics.n_latent_pairs}S{cfg.geodesics.steps}LR{cfg.geodesics.lr}ND{cfg.num_decoders}.png"
    plt.title(f"Geodesics with {cfg.num_decoders} decoders in Latent space")
    plt.xlabel("Latent dimension 1")
    plt.ylabel("Latent dimension 2")
    plt.legend([pg_plot, sl_plot], ["Pullback Geodesic", "Straight Line"], handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.savefig(filename)
    print("Saved figure to", filename)
    # plt.show()
    

    
    
if __name__ == "__main__":
    main()