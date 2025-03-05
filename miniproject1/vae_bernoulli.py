import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
import torch.optim as optim
from torchvision import datasets, transforms
from torch.nn import functional as F
from tqdm import tqdm

# ---------- PRIOR -------------
class MoGPrior(nn.Module):
    def __init__(self, latent_dim=32, n_components=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.logits = nn.Parameter(torch.zeros(n_components))
        self.means = nn.Parameter(torch.randn(n_components, latent_dim))
        self.log_stds = nn.Parameter(torch.zeros(n_components, latent_dim))

    def forward(self):
        comp = td.Independent(td.Normal(self.means, torch.exp(self.log_stds)), 1)
        mix = td.Categorical(logits=self.logits)
        return td.MixtureSameFamily(mix, comp)

# ---------- ENCODER -------------
class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        super().__init__()
        self.encoder_net = encoder_net
        
    def forward(self, x):
        # encoder_net returns [batch_size, 2*latent_dim], chunk to get mean, log_std
        mean, log_std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(mean, torch.exp(log_std)), 1)

# ---------- DECODER -------------
class GaussianDecoderFixedVar(nn.Module):
    def __init__(self, decoder_net, sigma=0.1):
        super().__init__()
        self.decoder_net = decoder_net
        self.sigma = sigma
        
    def forward(self, z):
        mu = self.decoder_net(z)  # shape: [batch_size, 784]
        mu = mu.view(-1, 28, 28)
        return td.Independent(td.Normal(loc=mu, scale=self.sigma), 2)

# ---------- VAE -------------
class VAE(nn.Module):
    def __init__(self, prior, encoder, decoder):
        super().__init__()
        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder
        
    def elbo(self, x):
        q = self.encoder(x)
        z = q.rsample()
        log_pxz = self.decoder(z).log_prob(x)
        log_pz = self.prior().log_prob(z)
        log_qz = q.log_prob(z)
        return torch.mean(log_pxz + log_pz - log_qz)

    def forward(self, x):
        # Negative ELBO
        return -self.elbo(x)

    def sample(self, num_samples=64):
        prior_dist = self.prior()
        z = prior_dist.sample((num_samples,))
        return self.decoder(z).sample()  # shape: [num_samples, 28, 28]

# ---------- MAIN SCRIPT EXAMPLE -------------
if __name__ == "__main__":
    # 1) Load continuous MNIST
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x.squeeze())  # shape: [28,28]
                       ])),
        batch_size=64, shuffle=True
    )

    # 2) Define networks
    latent_dim = 32
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 2*latent_dim)  # mean + log_std
    )
    decoder_net = nn.Sequential(
        nn.Linear(latent_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784)
    )

    # 3) Define prior, encoder, decoder, VAE
    prior = MoGPrior(latent_dim=latent_dim, n_components=5)
    encoder = GaussianEncoder(encoder_net)
    decoder = GaussianDecoderFixedVar(decoder_net, sigma=0.1)
    model = VAE(prior, encoder, decoder)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 4) Train
    epochs = 5
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, _ in tqdm(train_loader):
            # x: [batch_size, 28,28], must be float
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # 5) (Optional) Generate samples
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples=64)  # shape [64, 28, 28]
        print("Sample shape:", samples.shape)
