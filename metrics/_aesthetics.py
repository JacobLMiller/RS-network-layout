import gdMetriX as gdm 
import numpy as np
import networkx as nx

def node_distribution(G: nx.Graph,X: np.ndarray):
    nx.set_node_attributes(G, {v: X[i] for i,v in enumerate(G.nodes())}, "pos")

    return gdm.concentration(G,'pos')

def angular_resolution(G:nx.Graph, X:np.ndarray):
    nx.set_node_attributes(G, {v: X[i] for i,v in enumerate(G.nodes())}, "pos")

    return gdm.angular_resolution(G,'pos')

def edge_length_variation(G:nx.Graph, X:np.ndarray):
    nx.set_node_attributes(G, {v: X[i] for i,v in enumerate(G.nodes())}, "pos")

    return gdm.edge_length_deviation(G,'pos')