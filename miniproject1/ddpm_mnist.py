# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(
            self.alpha.cumprod(dim=0), requires_grad=False
        )

    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """

        ### Implement Algorithm 1 here ###
        sqrt_alpha_bar = torch.sqrt(
            self.alpha_cumprod
        )  # HINT: use torch.sqrt to calculate the sqrt of alphas_bar at timestep t

        noise = torch.normal(0, 1, size=x.shape)
        t = torch.randint(1, self.T, (x.shape[0],))
        thing = (
            sqrt_alpha_bar[t].unsqueeze(1) * x
            + torch.sqrt(1 - sqrt_alpha_bar[t]).unsqueeze(1) * noise
        )
        pred_noise = self.network(thing, t.unsqueeze(1).float())
        neg_elbo = torch.mean((noise - pred_noise) ** 2)

        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape).to(self.alpha.device)

        # Sample x_t given x_{t+1} until x_0 is sampled
        for t in range(self.T - 1, -1, -1):
            ### Implement the remaining of Algorithm 2 here ###
            z = torch.normal(torch.zeros_like(x_t), 1)
            z[t < 1] = 0
            std = torch.sqrt(self.beta)
            predicted_noise = self.network(x_t, t)
            x_t = (
                1
                / torch.sqrt(self.alpha)
                * (
                    x_t
                    - (1 - self.alpha)
                    / (torch.sqrt(1 - self.alpha_cumprod))
                    * predicted_noise
                )
                + std * z
            )  # Calculate x_{t-1}, see line 4 of the Algorithm 2 (Sampling) at page 4 of the ddpm paper.

        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(
                loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}"
            )
            progress_bar.update()


class FcNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden):
        """
        Initialize a fully connected network for the DDPM, where the forward function also take time as an argument.

        parameters:
        input_dim: [int]
            The dimension of the input data.
        num_hidden: [int]
            The number of hidden units in the network.
        """
        super(FcNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, input_dim),
        )

    def forward(self, x, t):
        """ "
        Forward function for the network.

        parameters:
        x: [torch.Tensor]
            The input data of dimension `(batch_size, input_dim)`
        t: [torch.Tensor]
            The time steps to use for the forward pass of dimension `(batch_size, 1)`
        """
        x_t_cat = torch.cat([x, t], dim=1)
        return self.network(x_t_cat)


if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "test"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="tg",
        choices=["tg", "cb", "mnist"],
        help="dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model.pt",
        help="file to save model to or load model from (default: %(default)s)",
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
        default=10000,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: %(default)s)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="V",
        help="learning rate for training (default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    mnist_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255),
            transforms.Lambda(lambda x: (x - 0.5) * 2.0),
            transforms.Lambda(lambda x: x.flatten()),
        ]
    )

    class MNISTWithoutLabels(datasets.MNIST):
        def __getitem__(self, index):
            data, target = super().__getitem__(index)
            return data

    # Generate the data
    target_class = 0
    mnist = MNISTWithoutLabels(
        "data", train=True, download=True, transform=mnist_transform
    )
    idx = torch.as_tensor(mnist.targets) == target_class
    mnist = torch.utils.data.dataset.Subset(mnist, np.where(idx == 1)[0])

    train_loader = torch.utils.data.DataLoader(
        mnist, batch_size=args.batch_size, shuffle=True
    )

    # Plot the first batch of data
    data_iter = iter(train_loader)
    x = next(data_iter)
    x = x.view(-1, 1, 28, 28)
    save_image(x, "data.png", nrow=5)

    # Get the dimension of the dataset
    D = next(iter(train_loader)).shape[1]

    # Define the network
    num_hidden = 64
    network = FcNetwork(D, num_hidden)

    # Set the number of steps in the diffusion process
    T = 1000

    # Define model
    model = DDPM(network, T=T).to(args.device)

    # Choose mode to run
    if args.mode == "train":
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        train(model, optimizer, train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == "sample":
        import matplotlib.pyplot as plt
        import numpy as np

        # Load the model
        model.load_state_dict(
            torch.load(args.model, map_location=torch.device(args.device))
        )

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample((100, D))).cpu()

        # Transform the samples back to the original space
        samples = samples / 2 + 0.5

        # Plot the density of the toy data and the model samples
        # Plot mnist samples
        samples = samples.view(-1, 1, 28, 28)
        print(f"{args.samples}Epoch{args.epochs}.png")
        save_image(samples, f"{args.samples}Epoch{args.epochs}Class{target_class}.png", nrow=10)
