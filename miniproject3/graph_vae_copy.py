import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
import networkx as nx
from datetime import datetime



class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, latent_dim)
        self.conv_logvar = GCNConv(hidden_channels, latent_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar

class MLPDecoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * latent_dim, latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, z):
        # Compute pairwise combinations of node embeddings
        num_nodes = z.size(0)
        z_i = z.unsqueeze(1).repeat(1, num_nodes, 1)  # Shape: [num_nodes, num_nodes, latent_dim]
        z_j = z.unsqueeze(0).repeat(num_nodes, 1, 1)  # Shape: [num_nodes, num_nodes, latent_dim]
        z_pairs = torch.cat([z_i, z_j], dim=-1)       # Shape: [num_nodes, num_nodes, 2 * latent_dim]

        # Decode edge probabilities
        adj_reconstructed = self.mlp(z_pairs).squeeze(-1)  # Shape: [num_nodes, num_nodes]
        return adj_reconstructed
    
class GraphVAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, edge_index, edge_index_true):
        mu, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logvar)
        adj_pred = self.decoder(z)
        
        # True adjacency
        adj_true = to_dense_adj(edge_index_true)[0]

        return adj_pred, adj_true, mu, logvar
    
    def sample(self, x, edge_index):
        """
        Samples a latent vector z from the encoder and returns it.
        """
        mu, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return z
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def elbo(self, adj_pred, adj_true, mu, logvar):
        recon_loss = F.binary_cross_entropy(adj_pred, adj_true)
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div

device = 'cpu'

# Load data
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
node_feature_dim = 7

# Split into training and validation
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

# Create dataloader for training and validation
train_loader = DataLoader(train_dataset, batch_size=100)
validation_loader = DataLoader(validation_dataset, batch_size=44)
test_loader = DataLoader(test_dataset, batch_size=44)

# Model
in_channels = dataset.num_node_features
hidden_channels = 32
latent_dim = 16

encoder = GCNEncoder(in_channels, hidden_channels, latent_dim)
decoder = MLPDecoder(latent_dim)
model = GraphVAE(encoder, decoder)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


model.train()



# Training loop
for epoch in range(401):
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        adj_pred, adj_true, mu, logvar = model(data.x, data.edge_index, data.edge_index)
        loss = model.elbo(adj_pred, adj_true, mu, logvar)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}')


        
        
# Evaluation
model.eval()
total_loss = 0
for data in test_loader:
    data = data.to(device)
    with torch.no_grad():
        adj_pred, adj_true, mu, logvar = model(data.x, data.edge_index, data.edge_index)
        loss = model.elbo(adj_pred, adj_true, mu, logvar)
        total_loss += loss.item()

        # Calculate accuracy
        adj_pred_binary = (adj_pred > 0.5).float()
        correct = (adj_pred_binary == adj_true).sum().item()
        total = adj_true.numel()
        accuracy = correct / total

    print(f'Validation Loss: {total_loss / len(validation_loader):.4f}, Accuracy: {accuracy:.4f}')

# Get current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Get current date
current_date = datetime.now().strftime("%Y-%m-%d")

# Save model with accuracy and date in the filename
model_accuracy = f"{accuracy:.4f}"
model_filename = f"graph_vae_model_acc_{model_accuracy}_date_{current_date}.pth"
model_filepath = f"models/{model_filename}"
torch.save(model.state_dict(), model_filename)
print(f"Model saved as {model_filename}")

# # Load the model
# model_filename = "graph_vae_model_acc_0.9973_datetime_2025-05-01_14-30-42.pth"
# model_filepath = f"models/{model_filename}"
# loaded_model = GraphVAE(encoder, decoder)
# loaded_model.load_state_dict(torch.load(model_filepath))
# print("Model loaded successfully.")

# Sample a graph
model.eval()
for data in test_loader:
    data = data.to(device)
    with torch.no_grad():
        print(data.x)
        print(data.edge_index)
        
        z = model.sample(data.x, data.edge_index)
        adj_pred = model.decoder(z)
        adj_pred_binary = (adj_pred > 0.5).float()
        print(adj_pred_binary.sum())
        
        

        # Convert predicted adjacency matrix to NetworkX graph
        pred_graph = nx.from_numpy_array(adj_pred_binary.cpu().numpy())
        
        # Plot the predicted graph
        plt.figure(figsize=(8, 8))
        nx.draw(pred_graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=100, font_size=10)
        plt.title("Sampled Graph (Predicted)")
        plt.show()

        # Convert true adjacency matrix to NetworkX graph
        adj_true = to_dense_adj(data.edge_index)[0]
        
        zero_matrix = torch.zeros_like(adj_pred_binary)
        correct = (zero_matrix == adj_true).sum().item()
        total = adj_true.numel()
        accuracy = correct / total
        
        print(f'Accuracy: {accuracy:.4f}')
        
        true_graph = nx.from_numpy_array(adj_true.cpu().numpy())

        # Plot the true graph
        plt.figure(figsize=(8, 8))
        nx.draw(true_graph, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=100, font_size=10)
        plt.title("True Graph")
        plt.show()
        break
