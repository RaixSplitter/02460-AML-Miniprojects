"""
viz_three_graphs.py
Visualise empirical, Erdős Rényi baseline, and Graph-VAE graphs.
-----------------------------------------------------------------
Requires:
• networkx
• matplotlib
• torch & torch_geometric  (only for the empirical MUTAG molecule)
• graph_vae.py + trained checkpoint 'graph_vae.pt'
"""

import os, random, numpy as np, networkx as nx, matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1) empirical MUTAG molecule (fallback: simple path graph)
# ---------------------------------------------------------------
try:
    from torch_geometric.datasets import TUDataset
    from torch_geometric.utils import to_networkx
    dataset = TUDataset(root="data", name="MUTAG")
    empirical = to_networkx(dataset[0]).to_undirected()          # pick the first molecule
except Exception as e:
    print("⚠︎ MUTAG not available – using a 10-node path graph instead.")
    empirical = nx.path_graph(10)

N = empirical.number_of_nodes()
M = empirical.number_of_edges()
density = (2*M)/(N*(N-1)) if N > 1 else 0.0

# ---------------------------------------------------------------
# 2) Erdős–Rényi sample with same N & density
# ---------------------------------------------------------------
from erdos_renyi_baseline import ErdosRenyiBaseline

# build a *tiny* training set containing just the empirical graph,
# so the baseline’s P(N) and r_N are defined:
er = ErdosRenyiBaseline(seed=42)
er.fit([empirical])
er_sample = er.sample_graph()           # draw ONE graph

# ---------------------------------------------------------------
# 3) Graph-VAE sample with same N
# ---------------------------------------------------------------
from graph_vae import GraphVAE
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
vae = GraphVAE(in_dim=7).to(device)           # 7 is MUTAG’s node-feature size
vae.load_state_dict(torch.load("graph_vae.pt", map_location=device))
vae.eval()

with torch.no_grad():
    vae_sample = vae.generate_graph(N, device=device)

# ---------------------------------------------------------------
# 4) Plot them side-by-side
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 4))

titles = ["Empirical graph",
          "Erdős Rényi sample",
          "Graph-VAE sample"]

for ax, G, title in zip(axes,
                        [empirical, er_sample, vae_sample],
                        titles):
    pos = nx.spring_layout(G, seed=1)
    nx.draw(G, pos, ax=ax, node_size=80, with_labels=False)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.show()
