# train_vae_undirected.py
# ---------------------------------------------------------
# Train the undirected-edge GraphVAE on MUTAG
# ---------------------------------------------------------
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_undirected
from torch_geometric.loader import DataLoader

from graph_vae import GraphVAE      # ← new model file

# ----------------- data ---------------------------------
dataset = TUDataset(root="data", name="MUTAG")

# 1-liner: drop duplicate directions (keeps only i < j)
for g in dataset:
    g.edge_index = to_undirected(g.edge_index, num_nodes=g.num_nodes,
                                 reduce='add')   # 'add' is fine for MUTAG

# simple 80/20 split ---------------------------------------------------------
idx    = torch.randperm(len(dataset))
split  = int(0.8 * len(dataset))
train_set, val_set = dataset[idx[:split]], dataset[idx[split:]]

train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=1)

# ----------------- model & optimiser --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

in_dim = dataset.num_node_features           # MUTAG: 7 one-hot atom types
model  = GraphVAE(in_dim).to(device)
optim  = torch.optim.Adam(model.parameters(), lr=1e-3)

# optional: linear β warm-up for the first 50 epochs
warmup_epochs = 50

# ----------------- training loop ------------------------
for epoch in range(1, 201):
    model.train()
    tot = recon = kl = 0.0

    beta = min(1.0, epoch / warmup_epochs)   # β ∈ [0,1]

    for data in train_loader:
        data = data.to(device)
        loss, r, k = model(data, beta=beta)  # ← pass β
        optim.zero_grad()
        loss.backward()
        optim.step()

        tot += loss.item();  recon += r.item();  kl += k.item()

    if epoch % 10 == 0:
        n = len(train_loader)
        print(f"Epoch {epoch:3d}  "
              f"L={tot/n:.4f}  Recon={recon/n:.4f}  KL={kl/n:.4f}  β={beta:.2f}")

    # ----- quick validation every 20 epochs --------------
    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            vloss = sum(model(d.to(device))[0].item() for d in val_loader)
        print(f"   > Val loss: {vloss / len(val_loader):.4f}")

torch.save(model.state_dict(), "graph_vae_undirected.pt")
