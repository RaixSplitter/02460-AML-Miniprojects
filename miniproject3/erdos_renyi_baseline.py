# baseline_er.py
from collections import Counter, defaultdict
import random
import numpy as np
import networkx as nx
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

class ErdosRenyiBaseline:
    """
    Baseline generator for Mini-project 3 (02460, DTU)

    • fit(train_graphs) collects the empirical distribution P(N)
      and the mean density r_N for every node count N that
      appears in the training set.

    • sample_graph() returns a single NetworkX graph drawn from
      the fitted distribution.

    • sample(n) draws n independent graphs and returns a list.
    """
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self._p_N: list[int] = []          # list of node counts (multiset) – for sampling
        self._r_N: dict[int, float] = {}   # mean density per node count

    # ---------- fitting ----------
    @staticmethod
    def _density(G: nx.Graph) -> float:
        """Density = |E| / [N·(N-1)/2]"""
        n = G.number_of_nodes()
        m = G.number_of_edges()
        return (2 * m) / (n * (n - 1)) if n > 1 else 0.0

    def fit(self, train_graphs: list[nx.Graph]):
        """Collect empirical P(N) and mean densities r_N."""
        # Gather densities for every distinct N
        dens_by_N: defaultdict[int, list[float]] = defaultdict(list)
        for G in train_graphs:
            N = G.number_of_nodes()
            dens_by_N[N].append(self._density(G))
            self._p_N.append(N)

        # Compute mean density r_N
        self._r_N = {N: float(np.mean(ds)) for N, ds in dens_by_N.items()}

        if not self._r_N:
            raise ValueError("No graphs supplied – cannot fit baseline.")

    # ---------- sampling ----------
    def _sample_N(self) -> int:
        """Draw N from the empirical multinomial."""
        return self.rng.choice(self._p_N)

    def sample_graph(self) -> nx.Graph:
        """Generate a single Erdős–Rényi graph G(N, r_N)."""
        N = self._sample_N()
        r = self._r_N[N]
        return nx.gnp_random_graph(N, r, seed=self.rng.randrange(2**32))

    def sample(self, n_graphs: int = 1) -> list[nx.Graph]:
        """Generate n_graphs independent samples."""
        return [self.sample_graph() for _ in range(n_graphs)]
    
    def save_model(self, path: str) -> None:
        """Save the model state to a file."""
        state = {
            "p_N": self._p_N,
            "r_N": self._r_N,
            "seed": self.rng.seed
        }
        np.savez_compressed(path, **state)

    def load_model(self, path: str) -> None:
        """Load the model state from a file."""
        state = np.load(path, allow_pickle=True)
        self._p_N = state["p_N"].tolist()
        self._r_N = state["r_N"].item()
        self.rng.seed = state["seed"].item()


# --------------------- tiny usage demo ---------------------
if __name__ == "__main__":
   

    dataset = TUDataset(root="data", name="MUTAG")
    idx_train = np.arange(150)  # ← example split (use yours!)
    train_graphs = [to_networkx(dataset[i]).to_undirected() for i in idx_train]

    # 2) Fit + sample
    er = ErdosRenyiBaseline(seed=42)
    er.fit(train_graphs)
    fake_graphs = er.sample(1000)
    print("Fitted model: ", er._r_N)
    print("Example generated graph:", fake_graphs[0])

    
    # 3) Save and load model
    er.save_model("erdos_renyi_model.npz")
    er2 = ErdosRenyiBaseline()
    er2.load_model("erdos_renyi_model.npz")
    print("Loaded model:", er2._r_N)
    print("Sampled graph from loaded model:", er2.sample_graph())