import nglpy
import networkx as nx
import numpy as np

def shape_faithfulness(G: nx.classes.graph.Graph, beta=1.0, **kwargs):
    node_to_idx = {element: idx for idx, element in enumerate(G.nodes())}

    aGraph = nglpy.EmptyRegionGraph(beta=beta, **kwargs)
    embedding = get_embedding(G)
    aGraph.build(embedding)

    beta_skeleton = aGraph.neighbors()

    score = 0
    for node in G.nodes():
        obs_nbrs = set(node_to_idx[nbr] for nbr in G[node])
        pred_nbrs = set(beta_skeleton[node_to_idx[node]])
        if obs_nbrs or pred_nbrs:
            score += len(obs_nbrs & pred_nbrs) / len(obs_nbrs | pred_nbrs)
    if len(G.nodes()):
        score /= len(G.nodes())

    return score



def get_embedding(G: nx.classes.graph.Graph):
    coords = []
    for node, data in G.nodes(data=True):
        coords.append([data['x'], data['y']])
    embedding = np.array(coords).astype('float64') # nglpy.EmptyRegionGraph.build does not work with float32
    return embedding