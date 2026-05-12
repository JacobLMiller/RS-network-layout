from sklearn.metrics import pairwise_distances
import networkx as nx
import numpy as np


def neighborhood_preservation(G: nx.classes.graph.Graph, r: int = 1):
    nodelist = list(G.nodes())
    X = get_embedding(G, nodelist)
    dist = pairwise_distances(X)

    ret = 0
    for n in nodelist:
        should_nbrs = {n,}
        for _ in range(r):
            for v in should_nbrs.copy():
                should_nbrs |= set(G.neighbors(v))
        should_nbrs.discard(n)

        k = len(should_nbrs)
        if k == 0: continue
        indices = np.argsort(dist[n])[1:k+1]
        emb_nbrs = set(nodelist[i] for i in indices)
        ret += len(should_nbrs & emb_nbrs) / len(should_nbrs | emb_nbrs)

    return ret / X.shape[0]



def get_embedding(G: nx.classes.graph.Graph, nodelist):
    coords = []
    for node in nodelist:
        x, y = G.nodes[node]['x'], G.nodes[node]['y']
        coords.append([x,y])
    embedding = np.array(coords)
    return embedding