# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from sklearn.decomposition import PCA
import random

class GaussianPrior(nn.Module):
    """
    Standard Gaussian prior: p(z) = N(0, I).
    """
    def __init__(self, M):
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(M), requires_grad=False)

    def forward(self):
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class MoGPrior(nn.Module):
    """
    Mixture of Gaussians prior:
      p(z) = sum_k pi_k * N(z; mu_k, exp(log_std_k)^2).
    """
    def __init__(self, M, n_components=5):
        super().__init__()
        self.M = M
        self.n_components = n_components
        # Trainable mixture logits, means, and log-std
        self.logits = nn.Parameter(torch.zeros(n_components))
        self.means = nn.Parameter(torch.randn(n_components, M))
        self.log_stds = nn.Parameter(torch.zeros(n_components, M))

    def forward(self):
        pi = td.Categorical(logits=self.logits)
        comp = td.Independent(td.Normal(self.means, torch.exp(self.log_stds)), 1)
        return td.MixtureSameFamily(pi, comp)


class VampPrior(nn.Module):
    """
    VampPrior: 
      p(z) = (1 / n_pseudos) * sum_j q(z | x_j^*),
    where {x_j^*} are "pseudo-inputs" learned in data space,
    and q is the same encoder used by the VAE.
    """
    def __init__(self, M, encoder, n_pseudos=10, input_shape=(1, 28, 28)):
        super().__init__()
        self.M = M
        self.encoder = encoder  
        self.n_pseudos = n_pseudos
        
        self.pseudo_inputs = nn.Parameter(
            0.5 * torch.rand(n_pseudos, *input_shape)
        )

    def forward(self):
        with torch.no_grad():
            qj = self.encoder(self.pseudo_inputs)
        means = qj.base_dist.loc       
        scales = qj.base_dist.scale    


        cat = td.Categorical(logits=torch.zeros(self.n_pseudos, device=means.device))
        comp = td.Independent(td.Normal(means, scales), 1)
        return td.MixtureSameFamily(cat, comp)

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        A Gaussian encoder distribution q(z|x).
        encoder_net outputs (mean, log_std).
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        mean, log_std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(log_std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Bernoulli decoder distribution p(x|z).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class VAE(nn.Module):
    """
    Variational Autoencoder with a general prior module.
    We'll compute the ELBO via:
      E_q[log p(x|z) + log p(z) - log q(z|x)]
    rather than using td.kl_divergence(q, p) to avoid unimplemented KL for Mixture.
    """
    def __init__(self, prior, decoder, encoder):
        super(VAE, self).__init__()
        self.prior = prior     # prior() returns a distribution p(z)
        self.decoder = decoder # p(x|z)
        self.encoder = encoder # q(z|x)

    def elbo(self, x):
        q = self.encoder(x)      
        z = q.rsample()          
        log_p_x_given_z = self.decoder(z).log_prob(x)
        log_p_z = self.prior().log_prob(z)
        log_q_z_given_x = q.log_prob(z)
        
        elbo = log_p_x_given_z + log_p_z - log_q_z_given_x
        return torch.mean(elbo)

    def forward(self, x):
        """
        Negative ELBO for training.
        """
        return -self.elbo(x)

    @torch.no_grad()
    def sample(self, n_samples=1):
        """
        Sample x ~ p(x|z), z ~ p(z).
        """
        z = self.prior().sample((n_samples,))
        return self.decoder(z).sample()

def train(model, optimizer, data_loader, epochs, device):
    model.train()
    progress_bar = tqdm(range(epochs), desc="Training")
    for epoch in range(epochs):
        for x, _ in data_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)  # negative ELBO
            loss.backward()
            optimizer.step()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", epoch=f"{epoch+1}/{epochs}")
        progress_bar.update()

@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Evaluates the test set ELBO (which approximates the log-likelihood).
    Returns the average ELBO over the test set.
    """
    model.eval()
    total_elbo = 0.0
    total_samples = 0
    for x, _ in data_loader:
        x = x.to(device)
        elbo_val = model.elbo(x)
        batch_size = x.size(0)
        total_elbo += elbo_val.item() * batch_size
        total_samples += batch_size
    avg_elbo = total_elbo / total_samples
    return avg_elbo

@torch.no_grad()
def get_samples(model, data_loader, device, max_points=5000):
    """
    Returns prior and aggregate posterior samples (as numpy arrays) from the model.
    Applies PCA to reduce to 2D if necessary.
    """
    model.eval()
    # Sample from prior
    pz = model.prior()
    z_prior = pz.sample((max_points,)) 
    # Sample from posterior
    z_list = []
    total = 0
    for x, _ in data_loader:
        x = x.to(device)
        qz = model.encoder(x)
        z_samp = qz.rsample()
        z_list.append(z_samp.cpu())
        total += z_samp.size(0)
        if total >= max_points:
            break
    z_post = torch.cat(z_list, dim=0)[:max_points]

    z_prior_np = z_prior.cpu().numpy()
    z_post_np = z_post.cpu().numpy()
    latent_dim = z_prior_np.shape[1]
    
    if latent_dim > 2:
        # Combine prior and posterior to get a common PCA transformation
        combined = np.concatenate([z_prior_np, z_post_np], axis=0)
        pca = PCA(n_components=2)
        combined_2d = pca.fit_transform(combined)
        z_prior_2d = combined_2d[:max_points]
        z_post_2d = combined_2d[max_points:]
    else:
        z_prior_2d = z_prior_np
        z_post_2d = z_post_np

    return z_prior_2d, z_post_2d

@torch.no_grad()
def save_samples(model, device, out_file="samples.png", n_samples=64):
    """
    Samples from the prior, decodes, and saves a grid of images to out_file.
    """
    model.eval()
    samples = model.sample(n_samples).cpu()  # shape [n_samples, 28, 28]
    grid = make_grid(samples.unsqueeze(1), nrow=8)  # (batch, 1, 28, 28)
    save_image(grid, out_file)

def plot_combined_by_type(prior_samples_dict, posterior_samples_dict, out_file="combined_plot.png"):
    """
    Plots a grid with 2 rows and columns equal to the number of prior types.
    Top row: Prior samples for each type.
    Bottom row: Aggregate posterior samples for each type.
    """
    colors = {"gaussian": "blue", "mog": "green", "vamp": "red"}
    types = list(prior_samples_dict.keys())
    n = len(types)
    fig, axs = plt.subplots(2, n, figsize=(5*n, 10))
    for i, pt in enumerate(types):
        axs[0, i].scatter(prior_samples_dict[pt][:, 0], prior_samples_dict[pt][:, 1],
                          alpha=0.4, s=10, color=colors[pt])
        axs[0, i].set_title(f"{pt.capitalize()} Prior")
        axs[1, i].scatter(posterior_samples_dict[pt][:, 0], posterior_samples_dict[pt][:, 1],
                          alpha=0.4, s=10, color=colors[pt])
        axs[1, i].set_title(f"{pt.capitalize()} Posterior")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"Combined plot saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'multi_run'],
                        help='mode to run: single run (train/sample) or multi_run for multiple seeds and combined evaluation')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save/load model')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable')
    parser.add_argument('--prior-type', type=str, default='gaussian', choices=['gaussian','mog','vamp'],
                        help='which prior to use for single run')
    parser.add_argument('--out-prefix', type=str, default='plot', help='prefix for saving prior vs posterior plots')
    parser.add_argument('--runs', type=int, default=3, help='number of runs for multi_run mode')
    
    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)
    
    device = torch.device(args.device)
    
    threshold = 0.5
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > threshold).float().squeeze())
    ])
    mnist_trainset = datasets.MNIST('data/', train=True, download=True, transform=transform)
    target_class = 0
    idx = torch.as_tensor(mnist_trainset.targets) == target_class
    mnist_trainset = torch.utils.data.dataset.Subset(mnist_trainset,np.where(idx == 1)[0])
    mnist_train_loader = torch.utils.data.DataLoader(
        mnist_trainset,
        batch_size=args.batch_size, shuffle=True
    )
    mnist_testset = datasets.MNIST('data/', train=True, download=True, transform=transform)
    target_class = 7
    idx = torch.as_tensor(mnist_testset.targets) == target_class
    mnist_testset = torch.utils.data.dataset.Subset(mnist_testset,np.where(idx == 1)[0])
    mnist_test_loader = torch.utils.data.DataLoader(
        mnist_testset,
        batch_size=args.batch_size, shuffle=False
    )
    
    M = args.latent_dim
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )
    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )
    
    if args.mode in ['train', 'sample']:
        # Choose prior type based on args.prior_type
        prior_type = args.prior_type
        if prior_type == 'gaussian':
            prior = GaussianPrior(M)
            prior_name = 'gaussian'
        elif prior_type == 'mog':
            prior = MoGPrior(M, n_components=5)
            prior_name = 'mog'
        elif prior_type == 'vamp':
            vamp_encoder_net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, M*2),
            )
            vamp_encoder = GaussianEncoder(vamp_encoder_net)
            prior = VampPrior(M, vamp_encoder, n_pseudos=10, input_shape=(1,28,28))
            prior_name = 'vamp'
        
        encoder = GaussianEncoder(encoder_net)
        decoder = BernoulliDecoder(decoder_net)
        model = VAE(prior, decoder, encoder).to(device)
        
        if args.mode == 'train':
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            train(model, optimizer, mnist_train_loader, args.epochs, device)
            torch.save(model.state_dict(), args.model)
            test_elbo = evaluate(model, mnist_test_loader, device)
            print(f"Test set ELBO (approximate log-likelihood): {test_elbo:.4f}")
            save_samples(model, device, out_file=args.samples)
        elif args.mode == 'sample':
            model.load_state_dict(torch.load(args.model, map_location=device))
            save_samples(model, device, out_file=args.samples)
            test_elbo = evaluate(model, mnist_test_loader, device)
            print(f"Test set ELBO (approximate log-likelihood): {test_elbo:.4f}")
    
    elif args.mode == "multi_run":
        seed = 0
        prior_types = ["gaussian", "mog", "vamp"]
        results = {pt: [] for pt in prior_types}
    
        prior_samples_dict = {}
        posterior_samples_dict = {}
        
        for pt in prior_types:
            
                # Set random seeds for reproducibility
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                
                # Build the model with the current prior
                if pt == 'gaussian':
                    prior = GaussianPrior(M)
                elif pt == 'mog':
                    prior = MoGPrior(M, n_components=5)
                elif pt == 'vamp':
                    vamp_encoder_net = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(784, 512),
                        nn.ReLU(),
                        nn.Linear(512, 512),
                        nn.ReLU(),
                        nn.Linear(512, M*2),
                    )
                    vamp_encoder = GaussianEncoder(vamp_encoder_net)
                    prior = VampPrior(M, vamp_encoder, n_pseudos=10, input_shape=(1,28,28))
                
                encoder = GaussianEncoder(encoder_net)
                decoder = BernoulliDecoder(decoder_net)
                model = VAE(prior, decoder, encoder).to(device)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                train(model, optimizer, mnist_train_loader, args.epochs, device)
                test_elbo = evaluate(model, mnist_test_loader, device)
                results[pt].append(test_elbo)
                
        
        prior_samples, posterior_samples = get_samples(model, mnist_test_loader, device)
        prior_samples_dict[pt] = prior_samples
        posterior_samples_dict[pt] = posterior_samples
        
        plot_combined_by_type(prior_samples_dict, posterior_samples_dict, out_file="combined_plot.png")
