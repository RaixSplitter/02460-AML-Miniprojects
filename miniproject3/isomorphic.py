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


if __name__ == "__main__":
    # Example usage
    G1 = nx.Graph()
    G2 = nx.Graph()

    # Add nodes and edges to the graphs
    G1.add_edges_from([(0, 1), (1, 2), (2, 0)])
    G2.add_edges_from([(3, 4), (4, 5), (5, 3)])

    # Check if the graphs are isomorphic
    print(is_isomorphic(G1, G2))  # Output: True or False based on the graph structure
    
    # Example where graphs are not isomorphic
    G3 = nx.Graph()
    G4 = nx.Graph()

    # Add nodes and edges to the graphs
    G3.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])  # A square
    G4.add_edges_from([(0, 1), (1, 2), (2, 0)])          # A triangle

    # Check if the graphs are isomorphic
    print(is_isomorphic(G3, G4))  # Output: False

    # Example where graphs have the same number of nodes but are not isomorphic
    G5 = nx.Graph()
    G6 = nx.Graph()

    # Add nodes and edges to the graphs
    G5.add_edges_from([(0, 1), (1, 2), (2, 3)])  # A path of 4 nodes
    G6.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])  # A square

    # Check if the graphs are isomorphic
    print(is_isomorphic(G5, G6))  # Output: False