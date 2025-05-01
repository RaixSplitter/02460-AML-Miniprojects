# graph_vae.py
# ---------------------------------------------------------
# GraphVAE for MUTAG – Advanced ML (02460, DTU)
# ---------------------------------------------------------
import math
import random
from itertools import combinations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
import networkx as nx

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------
class GraphVAE(nn.Module):
    """
    * Node-level latent VAE
    * Encoder: 2-layer GCN ⇒ μ, logσ  per node
    * Decoder: Inner-product (σ(zᵤ · z_v))   –  same as Kipf & Welling 2016
    """
    def __init__(self, in_dim: int, hid_dim: int = 64, z_dim: int = 32):
        super().__init__()
        # -------- encoder ----------
        self.gcn1 = GCNConv(in_dim, hid_dim)
        self.gcn_mu = GCNConv(hid_dim, z_dim)
        self.gcn_logvar = GCNConv(hid_dim, z_dim)

    # -------- decoder ----------
    @staticmethod
    def decode(z, edge_index_pos, num_nodes):
        """
        Return logits for given *positive* edges and an equal-sized set
        of *negative* edges sampled on the fly.
        """
        # positive logits ------------------------------
        pos_logits = (z[edge_index_pos[0]] * z[edge_index_pos[1]]).sum(dim=1)

        # sample equally many negatives ----------------
        idx = set(range(num_nodes))
        pos_set = {tuple(e) for e in edge_index_pos.t().tolist()}
        neg_edges = []
        while len(neg_edges) < edge_index_pos.size(1):
            u, v = random.sample(idx, 2)
            if (u, v) not in pos_set and (v, u) not in pos_set:
                neg_edges.append((u, v))
        neg_edges = torch.tensor(neg_edges, device=z.device).t()
        neg_logits = (z[neg_edges[0]] * z[neg_edges[1]]).sum(dim=1)

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([torch.ones_like(pos_logits),
                            torch.zeros_like(neg_logits)], dim=0)
        return logits, labels

    # -------- VAE forward pass ----------
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = F.relu(self.gcn1(x, edge_index))
        mu = self.gcn_mu(h, edge_index)
        logvar = self.gcn_logvar(h, edge_index)

        # reparametrisation trick
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        logits, labels = self.decode(z, edge_index, data.num_nodes)

        # losses
        recon_loss = F.binary_cross_entropy_with_logits(logits, labels)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl, recon_loss.detach(), kl.detach()

    # -------------------------------------------------
    # Generation – Section 2 . 3 & 2 . 4
    # -------------------------------------------------
    def generate_graph(self, N: int, thresh: float = 0.5, device="cpu"):
        """
        Sample a *new* graph with N nodes from the prior p(z)=𝒩(0,I)
        and the inner-product decoder.
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(N, self.gcn_mu.out_channels, device=device)
            probs = torch.sigmoid(z @ z.t()).cpu().numpy()
            G = nx.Graph()
            G.add_nodes_from(range(N))
            for i, j in combinations(range(N), 2):
                if probs[i, j] > thresh:
                    G.add_edge(i, j)
        return G
