import numpy as np
import networkx as nx

def neighborhood_radii(arr, neighbors):
    """
    arr : np.ndarray
        array of shape (n_samples, n_dimensions)

    neighbors : list of np.ndarrays
        list of length n_samples, each entry being another list of numpy arrays of shape (n_dimensions,)
    """
    radii = []
    for vec, neighbors in zip(arr, neighbors):
        radii += max(vec - neighbor for neighbor in neighbors)

    return radii

def neighborhood_radius(G: nx.classes.graph.Graph, type1: str):
    radii = []
    for type1node, data in G.nodes(data=True):
        if data['type'] == type1:
            pos1 = np.array([data['x'], data['y']])
            neighbors = []
            for type2node in G[type1node]:
                data2 = G.nodes[type2node]
                pos2 = [data2['x'], data2['y']]
                neighbors.append(pos2)
            neighbors = np.array(neighbors)
            dist = np.linalg.norm(neighbors - pos1, axis=1)
            radii.append(np.max(dist))
    return max(radii)

def max_nbr_radius(G: nx.classes.graph.Graph):
    """
    Maximum edge length in the embedding.
    """
    dists = list()
    for n1, n2 in G.edges():
        data1, data2 = G.nodes[n1], G.nodes[n2]
        pos1 = np.array([data1['x'], data1['y']])
        pos2 = np.array([data2['x'], data2['y']])
        dist = np.linalg.norm(pos1 - pos2)
        dists.append(dist)
    return max(dists)

def avg_edge_length(G: nx.classes.graph.Graph):
    """
    Average edge length in the embedding.
    """
    dists = list()
    for n1, n2 in G.edges():
        data1, data2 = G.nodes[n1], G.nodes[n2]
        pos1 = np.array([data1['x'], data1['y']])
        pos2 = np.array([data2['x'], data2['y']])
        dist = np.linalg.norm(pos1 - pos2)
        dists.append(dist)
    return np.mean(dists)

def avg_nbr_radius(G: nx.classes.graph.Graph, type1: str):
    """
    Mean neighborhood radius of type1 nodes, where neighborhood radius is the minimum
    radius of the circle around a type1 node needed to cover all points in the 2D embedding.
    """
    radii = []
    for type1node, data in G.nodes(data=True):
        if data['type'] == type1:
            pos1 = np.array([data['x'], data['y']])
            neighbors = []
            for type2node in G[type1node]:
                data2 = G.nodes[type2node]
                pos2 = [data2['x'], data2['y']]
                neighbors.append(pos2)
            neighbors = np.array(neighbors)
            nbr_radius = np.linalg.norm(neighbors - pos1, axis=1)
            radii.append(np.max(nbr_radius))
    return np.mean(radii)

def nbr_radii(G: nx.classes.graph.Graph, type1: str):
    """
    List of neighborhood radii of nodes of 'type'=type1, where neighborhood radius is the minimum
    radius of the circle around a type1 node needed to cover all points in the 2D embedding
    """
    radii = []
    for type1node, data in G.nodes(data=True):
        if data['type'] == type1:
            pos1 = np.array([data['x'], data['y']])
            nbrs_pos = []
            for type2node in G[type1node]:
                data2 = G.nodes[type2node]
                pos2 = [data2['x'], data2['y']]
                nbrs_pos.append(pos2)
            nbrs_pos = np.array(nbrs_pos)
            nbr_radius = np.linalg.norm(nbrs_pos - pos1, axis=1)
            radii.append(np.max(nbr_radius))
    return (radii)