# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from copy import deepcopy
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from geodesics import *
from tqdm import tqdm
import random
import statistics


class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()

        elbo = torch.mean(
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )
        return elbo

    def get_latent(self, x):
        return self.encoder(x).rsample()

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """

    num_steps = len(data_loader) * epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                model = model
                optimizer.zero_grad()
                # from IPython import embed; embed()
                loss = model(x)
                loss.backward()
                optimizer.step()

                # Report
                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"total epochs ={epoch}, step={step}, loss={loss:.1f}"
                    )

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(
                    f"Stopping training at total epoch {epoch} and current loss: {loss:.1f}"
                )
                break


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics", "train_ensemble", "evaluate_ensemble"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiment",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples.png",
        help="file to save samples in (default: %(default)s)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=3,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=10,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=20,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )
    parser.add_argument(
        "--seeds",  # Seeds used to train ensembles
        type=int,  # So that inputs are parsed as integers
        nargs='+',
        default=[42, 0, 123, 1234, 1, 2, 3, 4, 5, 6],
        metavar="seeds",
        help="Different seeds to train the VAEs (default: %(default)s)",
    )


    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

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
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # Define prior distribution
    M = args.latent_dim

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

    def coefficient_of_variation(distances):
        dist_mean = statistics.mean(distances)
        dist_std  = statistics.stdev(distances)
        return dist_std / dist_mean

    # Choose mode to run
    if args.mode == "train":

        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(
            model,
            optimizer,
            mnist_train_loader,
            args.epochs_per_decoder,
            args.device,
        )
        os.makedirs(f"{experiments_folder}", exist_ok=True)

        torch.save(
            model.state_dict(),
            f"{experiments_folder}/model.pt",
        )

    elif args.mode == "train_ensemble":
        if args.seeds == None:
            Exception("Seeds for the ensemble needs to be given as list")
        for seed in args.seeds:
            print(args.seeds)
            print(seed)
            seed = int(seed)

            torch.manual_seed(seed)
            experiments_folder = f"{args.experiment_folder}_{seed}"
            os.makedirs(f"{experiments_folder}", exist_ok=True)
            for i in range(args.num_decoders):
                model = VAE(
                    GaussianPrior(M),
                    GaussianDecoder(new_decoder()),
                    GaussianEncoder(new_encoder()),
                ).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                train(
                    model,
                    optimizer,
                    mnist_train_loader,
                    args.epochs_per_decoder,
                    args.device,
                )
                os.makedirs(f"{experiments_folder}", exist_ok=True)
                torch.save(
                    model.state_dict(),
                    f"{experiments_folder}/model_decoder_{i}.pt",
                )

    elif args.mode == "sample":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)

            data = next(iter(mnist_test_loader))[0].to(device)
            recon = model.decoder(model.encoder(data).mean).mean
            save_image(
                torch.cat([data.cpu(), recon.cpu()], dim=0), "reconstruction_means.png"
            )

    elif args.mode == "eval":
        # Load trained model
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)

    elif args.mode == "geodesics":
        # Conf
        N_POINTS = None  # Number of latents to sample
        POINT_RESOLUTION = 20  # Number of points to sample along the geodesic
        N_LATENT_PAIRS = 35  # Number of latent pairs to sample
        STEPS = 1000 # Number of steps to optimize the geodesic, 1000 recommended with Monto Carlo
        GEODESIC_LEARNING_RATE = 0.1
        TQDM_DISABLE = False # False to show progress bar, True to disable it
        MONTO_MODE = True # True to use Monte Carlo, False to use the metric

        # Init model
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(
            torch.load(args.experiment_folder + "/model.pt", weights_only=True)
        )
        model.eval()  # Set layer behavior to eval mode, note that it doesn't disable gradient calculation

        # Freeze the model parameters
        model.decoder.requires_grad = False
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

                if not N_POINTS:  # If N_POINTS is not set, we want to sample all points
                    continue

                if len(all_latents) * args.batch_size >= N_POINTS:  # If we have enough points, break
                    break

        # Concatenate all latents and labels
        latent = torch.cat(all_latents, dim=0)
        labels = torch.cat(all_labels, dim=0)
        if N_POINTS:  # Limit to N_POINTS
            latent = latent[:N_POINTS]
            labels = labels[:N_POINTS]

        # Get latent pairs
        n = latent.shape[0]

        chosen_idx_pairs = random.choices(range(n), k=N_LATENT_PAIRS * 2)
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
        pbar = tqdm(total=len(chosen_pairs) * STEPS, desc="Geodesics", disable = TQDM_DISABLE)

        for i, (z_start, z_end) in enumerate(chosen_pairs):
            samples = np.linspace(0, 1, 30)
            path_init = [(1 - t) * z_start + t * z_end for t in samples]
            path_init = torch.stack(path_init).to(device)  # shape [30, 2]

            path_z = torch.nn.Parameter(path_init.clone(), requires_grad=True)

            optimizer = torch.optim.Adam([path_z], lr=GEODESIC_LEARNING_RATE)

            pbar.set_description(f"Geodesics: Starting optimization")
            for step_i in range(STEPS):
                optimizer.zero_grad()

                # Fix start and end points by assigning them directly without gradients
                with torch.no_grad():
                    path_z.data[0] = z_start.detach()
                    path_z.data[-1] = z_end.detach()

                # E = sum_{i} (z_{i+1} - z_i)^T g(z_i) (z_{i+1} - z_i)
                if MONTO_MODE:
                    mont_str = "MonteC"
                    E = energy_curve_monte_carlo(path_z, model.decoder)
                else:
                    mont_str = "Metric"
                    E = energy_curve_with_metric(path_z, model.decoder)
                E.backward()

                optimizer.step()
                pbar.set_description(f"Geodesics: {step_i+1}/{STEPS} | Latent pair {i+1}/{N_LATENT_PAIRS}")
                pbar.update(1)

            with torch.no_grad():
                path_z.data[0] = z_start
                path_z.data[-1] = z_end

            geodesic_coords = path_z.detach().cpu().numpy()
            plt.plot(geodesic_coords[:, 0], geodesic_coords[:, 1], "-b")
            geodesic_coords_saved.append(geodesic_coords)

        plt.scatter(latent[:, 0].cpu(), latent[:, 1].cpu(), c=labels, cmap="viridis", alpha=0.5)
        plt.savefig(f"results/geo{mont_str}PR{POINT_RESOLUTION}LP{N_LATENT_PAIRS}S{STEPS}LR{GEODESIC_LEARNING_RATE}.png")
        plt.show()
        

        geodesic_coords_saved = np.array(geodesic_coords_saved)
        np.save("geodesics.npy", geodesic_coords_saved)

    elif args.mode == "evaluate_ensemble":
        """
        Evaluate how CoV changes for 1,2,3 decoders across multiple seeds.
        We'll show how to compute distances (Eucl & Geo) and get CoV,
        and how to do a final plot (CoV vs #decoders).
        """

        # We'll store CoV results for each #dec in [1,2,3].
        # If you like, you can just run your script 3 times with different --num-decoders,
        # but let's assume we want to do all in one go:
        #   python main.py evaluate_ensemble --seeds 0 1 ... 9
        # We'll just loop over dec_list below:
        dec_list = [1, 2, 3]

        # For reproducibility of which test pairs we pick
        random.seed(999)

        data_iter = iter(mnist_test_loader)
        x_full, y_full = next(data_iter)  # just one batch for demonstration
        # pick 10 random pairs
        pairs_idx = random.sample(range(len(x_full)), 20)  # 20 so we form 10 pairs
        pairs_idx = [(pairs_idx[i], pairs_idx[i+1]) for i in range(0, 20, 2)]

        # We'll store the final average CoVs for plotting
        eucl_covs_for_plot = []
        geo_covs_for_plot = []

        for num_decs in dec_list:
            all_eucl = []
            all_geo  = []
            
            for seed in args.seeds:
                experiment_folder_seed = f"{args.experiment_folder}_{seed}"

                # 1) Load the encoder from e.g. "model_decoder_0.pt"
                #    (assuming that file also has the encoder stored in it)
                model = VAE(
                    GaussianPrior(args.latent_dim),
                    GaussianDecoder(new_decoder()),
                    GaussianEncoder(new_encoder())
                ).to(device)

                model.load_state_dict(
                    torch.load(f"{experiment_folder_seed}/model_decoder_0.pt", map_location=device),
                    strict=False
                )
                model.eval()

                # 2) Load all decoders for this ensemble
                decoders = []
                for d_i in range(num_decs):
                    dec = GaussianDecoder(new_decoder()).to(device)
                    dec.load_state_dict(
                        torch.load(f"{experiment_folder_seed}/model_decoder_{d_i}.pt", map_location=device),
                        strict=False
                    )
                    dec.eval()
                    decoders.append(dec)

                # Distances for each pair => lists
                eucl_dists_for_seed = []
                geo_dists_for_seed = []

                for (idx_a, idx_b) in pairs_idx:
                    ya = x_full[idx_a:idx_a+1].to(device)
                    yb = x_full[idx_b:idx_b+1].to(device)

                    with torch.no_grad():
                        za = model.encoder(ya).mean.squeeze(0)  # shape (latent_dim,)
                        zb = model.encoder(yb).mean.squeeze(0)

                    # Euclidean distance
                    eucl_dist = torch.norm(za - zb).item()

                    # Geodesic distance using ensemble
                    # (We can either interpret the final_energy as a distance or sqrt of that.)
                    path_opt, final_energy = find_ensemble_geodesic(
                        z_start=za, z_end=zb, decoders=decoders,
                        steps=200, lr=0.01, path_points=20, n_samples=5
                    )
                    # For the assignment, you might consider final_energy as "the geodesic distance",
                    # or you might do math.sqrt(final_energy). The instructions hint to use the sum of
                    # squared differences as your measure. We'll just store final_energy here.
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