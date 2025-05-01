import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
def is_isomorphic(G1, G2):
    """
    Check if two graphs are isomorphic using NetworkX given the Weisfeiler-lehman algorithm.
    """
    # Check if the graphs have the same number of nodes and edges
    if G1.number_of_nodes() != G2.number_of_nodes() or G1.number_of_edges() != G2.number_of_edges():
        return False

    # Compute the Weisfeiler-Lehman graph hashes for both graphs
    hash1 = weisfeiler_lehman_graph_hash(G1)
    hash2 = weisfeiler_lehman_graph_hash(G2)

    # Compare the hashes
    return hash1 == hash2