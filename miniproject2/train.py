import hydra
from models import EnsembleVAE, GaussianPrior, GaussianDecoder, GaussianEncoder, train
import os
import random
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.nn as nn


from omegaconf import DictConfig


def train_experiment(cfg, SEED, n_decoders = None):
    device = cfg.device
    n_decoders = cfg.n_decoders if n_decoders is None else n_decoders 
    
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

    # Choose mode to run

    experiments_folder = cfg.experiment_folder
    os.makedirs(f"{experiments_folder}", exist_ok=True)
    model = EnsembleVAE(
        GaussianPrior(M),
        GaussianDecoder(new_decoder()),
        GaussianEncoder(new_encoder()),
        n_decoders,
    ).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(
        model,
        optimizer,
        mnist_train_loader,
        cfg.epochs_per_decoder,
        cfg.device,
    )
    
    os.makedirs(f"{experiments_folder}/ND{n_decoders}", exist_ok=True)

    torch.save(
        model.state_dict(),
        f"{experiments_folder}/ND{n_decoders}/modelSeed{SEED}nD{n_decoders}.pt",
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.seed == 'random':
        for i in range(cfg.num_reruns):
            SEED = random.randint(0, 1000)
            for n_decoders in range(1, cfg.num_decoders+1):
                train_experiment(cfg, SEED, n_decoders)
        return
    
    train_experiment(cfg, cfg.seed)
    
if __name__ == "__main__":
    main()