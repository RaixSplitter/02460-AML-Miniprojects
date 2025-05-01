# graph_vae_undirected.py
# ---------------------------------------------------------
# GraphVAE with undirected‑edge support & GNN decoder
# Advanced ML (02460, DTU) – May 2025 rewrite (v2)
# ---------------------------------------------------------
from itertools import combinations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
import networkx as nx

# ---------------------------------------------------------
# Edge‑level decoder
# ---------------------------------------------------------
class EdgeDecoder(nn.Module):
    """A tiny GNN that takes latent nodes *z* and predicts logits for a set
    of edges (u,v).  We **separate** the graph used for message passing
    (``edge_index_msg`` – normally just the positive edges that define the
    molecule) from the list of edges we actually want a probability for
    (``edge_pairs`` = positives ∪ negatives).  The two can differ – in fact
    at training time we pass *only* the ground‑truth bonds for message
    passing but evaluate the MLP on both the positive and sampled negative
    edges.
    """

    def __init__(self, z_dim: int, hidden: int = 64, msg_passes: int = 2):
        super().__init__()
        self.convs = nn.ModuleList(GCNConv(z_dim, z_dim)
                                   for _ in range(msg_passes))
        self.mlp = nn.Sequential(
            nn.Linear(4 * z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))  # → logit

    def forward(self, z, edge_index_msg, edge_pairs):
        h = z
        for conv in self.convs:
            h = F.relu(conv(h, edge_index_msg))

        h_u = h[edge_pairs[0]]
        h_v = h[edge_pairs[1]]
        phi = torch.cat([h_u, h_v,
                         torch.abs(h_u - h_v),
                         h_u * h_v], dim=-1)
        return self.mlp(phi).squeeze(-1)


# ---------------------------------------------------------
# Variational Auto‑Encoder
# ---------------------------------------------------------
class GraphVAE(nn.Module):
    """Node‑level VAE that **expects an undirected single‑copy edge list**.

    *Encoder*  – 2‑layer GCN → µᵢ and logσ²ᵢ per node.
    *Decoder*  – EdgeDecoder above (GNN + MLP) instead of plain inner product.
    """

    def __init__(self, in_dim: int, hid_dim: int = 64, z_dim: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        # ------- encoder ---------------------------------
        self.gcn1 = GCNConv(in_dim, hid_dim, add_self_loops=False,
                            normalize=False)
        self.ln1 = nn.LayerNorm(hid_dim)
        self.gcn_mu = GCNConv(hid_dim, z_dim, add_self_loops=False,
                              normalize=False)
        self.gcn_logvar = GCNConv(hid_dim, z_dim, add_self_loops=False,
                                  normalize=False)
        self.dropout_p = dropout

        # ------- decoder ---------------------------------
        self.decoder = EdgeDecoder(z_dim)
        # learnable global temperature (helps with sparsity calibration)
        self.tau = nn.Parameter(torch.tensor(1.0))

    # -----------------------------------------------------
    # utilities
    # -----------------------------------------------------
    @staticmethod
    def _unique_pos_edges(edge_index):
        """Return the *i < j* subset so each undirected bond appears once."""
        mask = edge_index[0] < edge_index[1]
        return edge_index[:, mask]

    # -----------------------------------------------------
    # main forward / ELBO
    # -----------------------------------------------------
    def forward(self, data, beta: float = 1.0, neg_ratio: int = 5):
        x, edge_index = data.x, data.edge_index

        # ----- encoder -----------------------------------
        h1 = F.relu(self.ln1(self.gcn1(x, edge_index)))
        h1 = F.dropout(h1, p=self.dropout_p, training=self.training)
        mu = self.gcn_mu(h1, edge_index)
        logvar = self.gcn_logvar(h1, edge_index)

        # reparameterisation ------------------------------
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

        # ----- build positive & negative edge lists ------
        pos_edge = self._unique_pos_edges(edge_index)
        neg_edge = negative_sampling(pos_edge, num_nodes=data.num_nodes,
                                     num_neg_samples=neg_ratio * pos_edge.size(1),
                                     method='sparse')
        # message‑passing graph uses *only* positive bonds
        edge_pairs = torch.cat([pos_edge, neg_edge], dim=1)

        # ----- decoder -----------------------------------
        logits = self.decoder(z, pos_edge, edge_pairs) / self.tau.clamp(min=1e-4)
        labels = torch.cat([torch.ones(pos_edge.size(1), device=logits.device),
                            torch.zeros(neg_edge.size(1), device=logits.device)])
        # class‑skew correction
        pos_weight = torch.as_tensor((labels == 0).sum() / (labels == 1).sum(),
                                     device=logits.device, dtype=logits.dtype)
        recon = F.binary_cross_entropy_with_logits(logits, labels,
                                                   pos_weight=pos_weight)

        # KL ----------------------------------------------
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon + beta * kl
        return loss, recon.detach(), kl.detach()

    # -----------------------------------------------------
    # generation
    # -----------------------------------------------------
    def generate_graph(self, N: int, target_density: float = None,
                       device: str = "cpu"):
        """Sample a **new** graph from the prior.  If *target_density* is
        provided we keep adding edges in descending probability order until we
        hit that global density (nice for mimicking the training set)."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(N, self.gcn_mu.out_channels, device=device)
            logits = (z @ z.t()) / self.tau.clamp(min=1e-4)
            probs = torch.sigmoid(logits).cpu()

            # upper‑triangular indices (i<j)
            iu, ju = torch.triu_indices(N, N, offset=1)
            p_flat = probs[iu, ju]

            if target_density is None:
                thresh = 0.5  # default
                keep = p_flat > thresh
            else:
                k = int(target_density * len(p_flat))
                _, idx_sorted = torch.topk(p_flat, k)
                keep = torch.zeros_like(p_flat, dtype=torch.bool)
                keep[idx_sorted] = True

            G = nx.Graph()
            G.add_nodes_from(range(N))
            for i, j, flag in zip(iu.tolist(), ju.tolist(), keep.tolist()):
                if flag:
                    G.add_edge(i, j)
        return G
