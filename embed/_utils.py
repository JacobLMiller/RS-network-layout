import networkx as nx
import numpy as np
from umap import UMAP
from sklearn.manifold import TSNE, MDS

def read_json_reprsentation(jsongraph,gname):
    G = nx.Graph()

    G.add_nodes_from([(v['id'], v | {'type': "rich"}) for v in jsongraph['rich']])

    G.add_nodes_from([(v['id'], v | {'type': "sparse"}) for v in jsongraph['sparse']])

    edges = [(u['id'], v[0], v[1])
              if isinstance(v, list)
              else (u['id'], v, 1)
              for u in jsongraph['rich'] for v in u['s_ids']]
    G.add_weighted_edges_from(edges)

    G.remove_nodes_from(list(nx.isolates(G)))

    G.graph['name'] = gname

    return G

def get_embedding(G: nx.classes.graph.Graph):
    coords = []
    for node, data in G.nodes(data=True):
        coords.append([data['x'], data['y']])
    embedding = np.asarray(coords)
    return embedding

def dimension_reduction(X:np.ndarray, alg: str, X_is_dist: bool) -> np.ndarray:
    '''
    Parameters
    ----------
    Y :
        distance matrix or high-dimensional vector space

    alg : {'UMAP', 'TSNE', 'MDS'}

    X_is_dist : bool
        is X a distance matrix

    Returns
    -------



    '''
    metric = 'precomputed' if X_is_dist else 'euclidean'
    if alg == "UMAP":
        return UMAP(n_neighbors=15,min_dist=1e-2, metric=metric,n_components=2).fit_transform(X)
    
    elif alg == "TSNE":
        return TSNE(perplexity=30, metric=metric).fit_transform(X)
    
    elif alg == "MDS":
        print(X.shape)
        return MDS(dissimilarity=metric).fit_transform(X)
    
def set_arr(X: np.ndarray,G: nx.classes.graph.Graph):
    pos = {v: X[i] for i,v in enumerate(G.nodes())}
    for v in pos:
        G.nodes[v]['x'] = pos[v][0]
        G.nodes[v]['y'] = pos[v][1]

def getType(G: nx.classes.graph.Graph, nodetype:str) -> nx.classes.graph.Graph:
    return G.subgraph([node for node, data in G.nodes(data=True) if data['type'] == nodetype])