import numpy as np
import networkx as nx

from typing import List
from pyvis.network import Network


def display_network(adj_matrix: np.ndarray, feature_names: List[str], output_path: str):
    """
    Display network using pyvis
    
    Args:
        adj_matrix: adjacency matrix
        feature_names: feature names
        output_path: output path
    """

    G = nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph)

    # set node labels
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['label'] = feature_names[i]

    # change edge color
    for edge in G.edges():
        w = G.edges[edge]['weight']
        if w > 0:
            color = 'green'
        else:
            color = 'red'
        G.edges[edge]['color'] = color

    # remove low weight edges
    for edge in list(G.edges()):
        w = G.edges[edge]['weight']
        if abs(w) < 0.5:
            G.remove_edge(*edge)

    nt = Network(directed=True)
    nt.from_nx(G, show_edge_weights=True)

    nt.show(output_path)
