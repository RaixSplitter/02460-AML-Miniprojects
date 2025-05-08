import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader
from erdos_renyi_baseline import ErdosRenyiBaseline
from graph_vae_mpnn import GraphVAE
import networkx as nx
import matplotlib.pyplot as plt
from isomorphic import is_isomorphic
from tqdm import tqdm
import random

# ------------ CONFIG ---------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_PATH    = "graph_vae_mpnn.pt" 
# ------------ data -----------------
dataset = TUDataset(root="data", name="MUTAG")


train_graphs = [to_networkx(i).to_undirected() for i in dataset]
node_counts = [g.number_of_nodes() for g in train_graphs]
# 2) Fit + sample
er = ErdosRenyiBaseline(seed=42)
er.load_model("erdos_renyi_model.npz")

er_graphs = er.sample(1000)

novel_graphs = [True] * len(er_graphs)
unique_graphs = [True] * len(er_graphs)

in_dim = dataset.num_node_features
vae    = GraphVAE(in_dim).to(DEVICE)
vae.load_state_dict(torch.load(WEIGHT_PATH))
vae.eval()

def sample_vae_graphs(n_samples: int) -> list[nx.Graph]:
    """Draw graphs whose node-count follows the training distribution."""
    with torch.no_grad():
        sizes   = random.choices(node_counts, k=n_samples)
        graphs  = []
        for N in sizes:
            g = vae.generate_graph(N=N, device=DEVICE)   # no thresh needed now
            graphs.append(g)
    return graphs

vae_graphs = sample_vae_graphs(1000)


def benchmark(graphs, model):
    for i in tqdm(range(len(graphs))):
        for train_graph in train_graphs:
            if is_isomorphic(graphs[i], train_graph):
                novel_graphs[i] = False
                break
        
        for j in range(i + 1, len(graphs)):
            if is_isomorphic(graphs[i], graphs[j]):
                unique_graphs[i] = False
                unique_graphs[j] = False
                break
            
            
        
    print(f"{model}: Novel graphs: ", sum(novel_graphs))
    print(f"{model}: Unique graphs: ", sum(unique_graphs))
    print(f"{model}: Total graphs: ", len(er_graphs))

benchmark(er_graphs, "Erdos Renyi")
benchmark(vae_graphs, "Graph VAE")
# Compute Average Node Degree, Clustering Coefficient, and Eigenvector Centrality
def compute_graph_statistics(graphs):
    avg_node_degrees = []
    clustering_coeffs = []
    eigenvector_centralities = []

    for graph in graphs:
        avg_node_degrees.append(sum(dict(graph.degree()).values()) / graph.number_of_nodes())
        clustering_coeffs.append(nx.average_clustering(graph))
        try:
            eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=100000)
            eigenvector_centralities.append(sum(eigenvector_centrality.values()) / len(eigenvector_centrality))
        except nx.NetworkXError:
            eigenvector_centralities.append(0)  # Handle cases where eigenvector centrality fails

    return avg_node_degrees, clustering_coeffs, eigenvector_centralities

def plot_graph_statistics(avg_node_degrees, clustering_coeffs, eigenvector_centralities, title_prefix):
    plt.figure(figsize=(15, 5))

    # Average Node Degree
    plt.subplot(1, 3, 1)
    plt.hist(avg_node_degrees, bins=20, color='blue', alpha=0.7)
    plt.title(f"{title_prefix}: Average Node Degree")
    plt.xlabel("Average Node Degree")
    plt.ylabel("Frequency")

    # Clustering Coefficient
    plt.subplot(1, 3, 2)
    plt.hist(clustering_coeffs, bins=20, color='green', alpha=0.7)
    plt.title(f"{title_prefix}: Clustering Coefficient")
    plt.xlabel("Clustering Coefficient")
    plt.ylabel("Frequency")

    # Eigenvector Centrality
    plt.subplot(1, 3, 3)
    plt.hist(eigenvector_centralities, bins=20, color='red', alpha=0.7)
    plt.title(f"{title_prefix}: Eigenvector Centrality")
    plt.xlabel("Eigenvector Centrality")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# Example usage
avg_node_degrees, clustering_coeffs, eigenvector_centralities = compute_graph_statistics(train_graphs)

print("Average Node Degree: ", sum(avg_node_degrees) / len(avg_node_degrees))
print("Average Clustering Coefficient: ", sum(clustering_coeffs) / len(clustering_coeffs))
print("Average Eigenvector Centrality: ", sum(eigenvector_centralities) / len(eigenvector_centralities))



# Example usage for train graphs
plot_graph_statistics(avg_node_degrees, clustering_coeffs, eigenvector_centralities, "Empirical Graphs")

avg_node_degrees, clustering_coeffs, eigenvector_centralities = compute_graph_statistics(er_graphs)

print("Average Node Degree: ", sum(avg_node_degrees) / len(avg_node_degrees))
print("Average Clustering Coefficient: ", sum(clustering_coeffs) / len(clustering_coeffs))
print("Average Eigenvector Centrality: ", sum(eigenvector_centralities) / len(eigenvector_centralities))

plot_graph_statistics(avg_node_degrees, clustering_coeffs, eigenvector_centralities, "Erdos Renyi Graphs")


avg_node_degrees, clustering_coeffs, eigenvector_centralities = compute_graph_statistics(vae_graphs)
print("Average Node Degree: ", sum(avg_node_degrees) / len(avg_node_degrees))
print("Average Clustering Coefficient: ", sum(clustering_coeffs) / len(clustering_coeffs))
print("Average Eigenvector Centrality: ", sum(eigenvector_centralities) / len(eigenvector_centralities))

plot_graph_statistics(avg_node_degrees, clustering_coeffs, eigenvector_centralities, "Graph VAE Graphs")