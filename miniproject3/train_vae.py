# train_vae.py
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader

from graph_vae import GraphVAE

# ------------ data -----------------
dataset = TUDataset(root="data", name="MUTAG")
# simple 80/20 split
idx = torch.randperm(len(dataset))
split = int(0.8*len(dataset))
train_set = dataset[idx[:split]]
val_set   = dataset[idx[split:]]

train_loader = DataLoader(train_set, batch_size=1, shuffle=True)  # 1-graph per batch
val_loader   = DataLoader(val_set,   batch_size=1)

in_dim = dataset.num_node_features   # MUTAG: 7-dim one-hot atom type
model = GraphVAE(in_dim).to("cpu")
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------ loop ------------------
for epoch in range(1, 201):
    model.train()
    tot, recon, kl = 0, 0, 0
    for data in train_loader:
        data = data.to("cpu")
        loss, r, k = model(data)
        optim.zero_grad(); loss.backward(); optim.step()
        tot += loss.item(); recon += r.item(); kl += k.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}  L={tot/len(train_loader):.4f} "
              f"Recon={recon/len(train_loader):.4f} KL={kl/len(train_loader):.4f}")

    # ---------- tiny validation ----------
    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            vloss = sum(model(d.to('cpu'))[0].item() for d in val_loader)
        print(f"   > Val loss: {vloss / len(val_loader):.4f}")

torch.save(model.state_dict(), "graph_vae.pt")
