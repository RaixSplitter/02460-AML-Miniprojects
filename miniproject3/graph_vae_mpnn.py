
from itertools import combinations
from math import comb
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import negative_sampling
import networkx as nx
import numpy as np


class MPNNEncoder(nn.Module):
    """Message-passing encoder that returns both node- and graph-level params."""

    def __init__(self, in_dim: int, state_dim: int = 64, z_dim: int = 32,
                 g_dim: int = 16, rounds: int = 6, dropout: float = 0.25):
        super().__init__()
        self.rounds = rounds
        self.state_dim = state_dim
        self.dropout = dropout
        self.z_dim = z_dim
        self.g_dim = g_dim

        self.input_net = nn.Sequential(
            nn.Linear(in_dim, state_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        self.message_net = nn.ModuleList([
            nn.Sequential(nn.Linear(state_dim, state_dim), nn.ReLU(), nn.Dropout(dropout))
            for _ in range(rounds)
        ])
        self.update_net = nn.ModuleList([
            nn.Sequential(nn.Linear(state_dim, state_dim), nn.ReLU(), nn.Dropout(dropout))
            for _ in range(rounds)
        ])
        self.gru = nn.GRUCell(state_dim, state_dim)

        # node‑level projections
        self.to_mu_node     = nn.Linear(state_dim, z_dim)
        self.to_logvar_node = nn.Linear(state_dim, z_dim)

        # graph‑level projections (2‑layer MLP)
        self.graph_mu     = nn.Sequential(nn.Linear(state_dim, state_dim), nn.ReLU(), nn.Linear(state_dim, g_dim))
        self.graph_logvar = nn.Sequential(nn.Linear(state_dim, state_dim), nn.ReLU(), nn.Linear(state_dim, g_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = self.input_net(x)  # (N, D)

        for r in range(self.rounds):
            m   = self.message_net[r](h)
            agg = torch.zeros_like(h)
            agg = agg.index_add(0, edge_index[1], m[edge_index[0]])
            msg = self.update_net[r](agg)
            h   = self.gru(msg, h)

        # node params
        mu_node     = self.to_mu_node(h)
        logvar_node = self.to_logvar_node(h)

        # graph params
        g_readout   = h.mean(dim=0, keepdim=True)         # (1,D)
        mu_g        = self.graph_mu(g_readout).squeeze(0)     # (g_dim,)
        logvar_g    = self.graph_logvar(g_readout).squeeze(0) # (g_dim,)

        return mu_node, logvar_node, mu_g, logvar_g

class EdgeDecoderMP(nn.Module):
    """MPNN edge decoder (unchanged except for exposed helpers)."""

    def __init__(self, state_dim: int = 64, rounds: int = 2, dropout: float = 0.25):
        super().__init__()
        self.rounds = rounds
        self.dropout = dropout
        self.state_dim = state_dim

        self.message_net = nn.ModuleList([
            nn.Sequential(nn.Linear(state_dim, state_dim), nn.ReLU(), nn.Dropout(dropout))
            for _ in range(rounds)
        ])
        self.update_net = nn.ModuleList([
            nn.Sequential(nn.Linear(state_dim, state_dim), nn.ReLU(), nn.Dropout(dropout))
            for _ in range(rounds)
        ])
        self.gru = nn.GRUCell(state_dim, state_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(4 * state_dim, state_dim), nn.ReLU(), nn.Linear(state_dim, 1)
        )

    def message_pass(self, h: torch.Tensor, edge_index: torch.Tensor):
        for r in range(self.rounds):
            m   = self.message_net[r](h)
            agg = torch.zeros_like(h)
            agg = agg.index_add(0, edge_index[1], m[edge_index[0]])
            msg = self.update_net[r](agg)
            h   = self.gru(msg, h)
        return h

    def score_pairs(self, h: torch.Tensor, edge_index: torch.Tensor):
        h_u, h_v = h[edge_index[0]], h[edge_index[1]]
        phi = torch.cat([h_u, h_v, (h_u-h_v).abs(), h_u*h_v], dim=-1)
        return self.edge_mlp(phi).squeeze(-1)

class GraphVAE(nn.Module):
    """Node- & graph-level VAE with MPNN decoder."""

    def __init__(
        self,
        in_dim: int,
        hid_dim: int = 64,
        z_dim: int   = 32,
        g_dim: int   = 16,
        rounds_enc: int = 6,
        rounds_dec: int = 2,
        dropout: float  = 0.25,
        max_nodes: int  = 30,
        pos_weight: float = 12.0,   # MUTAG default
    ):
        super().__init__()
        self.z_dim = z_dim
        self.g_dim = g_dim

        self.encoder = MPNNEncoder(in_dim, hid_dim, z_dim, g_dim, rounds_enc, dropout)
        self.decoder = EdgeDecoderMP(hid_dim, rounds_dec, dropout)

        self.latent_proj = nn.Linear(z_dim + g_dim, hid_dim)

        self.tau        = nn.Parameter(torch.tensor(1.0))
        self.logit_bias = nn.Parameter(torch.zeros(()))

        self.register_buffer('pos_weight', torch.tensor(pos_weight))

        self.max_nodes = max_nodes

    def _edge_logits(self, z_cat: torch.Tensor, edge_ij: torch.Tensor, edge_pos: torch.Tensor):
        h0 = F.relu(self.latent_proj(z_cat))
        h  = self.decoder.message_pass(h0, edge_pos)         # only positives
        raw = self.decoder.score_pairs(h, edge_ij) / self.tau
        return raw + self.logit_bias

    def decode(self, z_cat: torch.Tensor, edge_pos: torch.Tensor, num_nodes: int, neg_ratio: int):
        neg_edge = negative_sampling(edge_pos, num_nodes=num_nodes,
                                     num_neg_samples=neg_ratio*edge_pos.size(1), method='sparse')
        all_edges = torch.cat([edge_pos, neg_edge], dim=1)

        logits = self._edge_logits(z_cat, all_edges, edge_pos)
        labels = torch.cat([torch.ones(edge_pos.size(1), device=z_cat.device),
                            torch.zeros(neg_edge.size(1),  device=z_cat.device)])
        return logits, labels

    def forward(self, data, beta=1.0, neg_ratio:int=12):
        mu_n, logvar_n, mu_g, logvar_g = self.encoder(data.x, data.edge_index)

        z_n = mu_n + torch.randn_like(mu_n) * torch.exp(0.5*logvar_n)   # (N,z)
        z_g = mu_g + torch.randn_like(mu_g) * torch.exp(0.5*logvar_g)   # (g_dim,)
        z_g_exp = z_g.unsqueeze(0).expand(z_n.size(0), -1)              # (N,g)
        z_cat = torch.cat([z_n, z_g_exp], dim=1)                        # (N,z+g)

        # recon
        logits, labels = self.decode(z_cat, data.edge_index, data.num_nodes, neg_ratio)
        recon = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=self.pos_weight)

        # KL (nodes + graph)
        kl_node  = -0.5 * torch.mean(1 + logvar_n - mu_n.pow(2) - logvar_n.exp())
        kl_graph = -0.5 * torch.mean(1 + logvar_g - mu_g.pow(2) - logvar_g.exp())
        loss = recon + beta * (kl_node + kl_graph)
        return loss, recon.detach(), (kl_node+kl_graph).detach()

    def _all_pairs_index(self, N: int, device):
        idx = torch.combinations(torch.arange(N, device=device), r=2)
        return idx.t()                               # (2, N·(N-1)//2)

    @torch.no_grad()
    def generate_graph(self, N: int, device="cpu"):
        """Sample a new graph with the full MPNN decoder (no hand-tuned threshold)."""
        assert N <= self.max_nodes
        self.eval()

        z_node = torch.randn(N,  self.z_dim, device=device)
        z_g    = torch.randn(1,  self.g_dim, device=device)      # single graph vector
        z_g    = z_g.repeat(N, 1)
        z_cat  = torch.cat([z_node, z_g], 1)                     # (N, z+g)

        h0     = F.relu(self.latent_proj(z_cat))                 # (N, state_dim)
        pairs  = self._all_pairs_index(N, device)
        logits = self.decoder.score_pairs(h0, pairs) / self.tau + self.logit_bias

        probs  = torch.sigmoid(logits).cpu().numpy()

        G = nx.Graph();  G.add_nodes_from(range(N))
        for (u, v), p in zip(pairs.t().cpu().numpy(), probs):
            if np.random.rand() < p:
                G.add_edge(u, v)

        # keep only the largest connected component
        if G.number_of_edges():
            largest_nodes = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_nodes).copy()
        return G

