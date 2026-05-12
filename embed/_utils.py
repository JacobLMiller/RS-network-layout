import json
import os

import networkx as nx
import numpy as np
from umap import UMAP
from sklearn.manifold import TSNE, MDS


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars and arrays to native Python types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

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

def dimension_reduction(X:np.ndarray, alg: str, X_is_dist: bool, n_neighbors: int = 15) -> np.ndarray:
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
        return UMAP(n_neighbors=n_neighbors, min_dist=1e-3, metric=metric, n_components=2).fit_transform(X)
    
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


def place_sparse_nodes(
    G: nx.Graph,
    rich_pos: dict,
    sparse_type: str = 'sparse',
    degree1_radius: float | None = None,
    seed: int | None = None,
) -> dict:
    """
    Compute 2-D positions for all sparse nodes.

    Sparse nodes with more than one rich neighbour are placed at the mean of those
    neighbours' positions. Degree-1 sparse nodes are placed at a small random offset
    from their single rich neighbour so they remain visually distinct from it.

    Parameters
    ----------
    G : nx.Graph
        Bipartite graph. Only edge structure is used here.
    rich_pos : dict
        Mapping of rich node id → 2-D numpy array, as returned by a layout method.
    sparse_type : str
        Value of the 'type' node attribute identifying sparse nodes.
    degree1_radius : float or None
        Offset distance for degree-1 nodes. None autoscales to 5 % of the mean
        per-axis standard deviation of the rich node positions.
    seed : int or None
        Random seed for reproducible degree-1 angle placement.

    Returns
    -------
    dict
        Mapping of sparse node id → 2-D numpy array.
    """
    rng = np.random.default_rng(seed)

    if degree1_radius is None and rich_pos:
        coords = np.array(list(rich_pos.values()))
        degree1_radius = 0.05 * float(np.std(coords, axis=0).mean())

    sparse_pos = {}
    for node, data in G.nodes(data=True):
        if data.get('type') != sparse_type:
            continue
        rich_nbrs = [n for n in G.neighbors(node) if n in rich_pos]
        if not rich_nbrs:
            continue
        if len(rich_nbrs) == 1:
            angle = rng.uniform(0.0, 2.0 * np.pi)
            offset = degree1_radius * np.array([np.cos(angle), np.sin(angle)])
            sparse_pos[node] = rich_pos[rich_nbrs[0]] + offset
        else:
            sparse_pos[node] = np.mean([rich_pos[n] for n in rich_nbrs], axis=0)

    return sparse_pos


def save_graph(
    G: nx.Graph,
    path: str,
    polygons: dict[str, dict[int, str]] | None = None,
    strip_keys: list[str] | None = None,
) -> None:
    """
    Serialise the graph and all computed layout data to a JSON file.

    Node positions ('x', 'y'), cluster IDs, and all other node attributes are
    written automatically from G. Polygon region strings produced by
    compute_cluster_polygons() are attached at the graph level under
    '{cluster_key}_polygons' as a semicolon-joined sequence ordered by cluster ID.

    Numpy scalars and arrays remaining in node attributes after the pipeline are
    serialised transparently. The 'vector' attribute added by preprocess() is
    stripped by default — it is large and not needed after layout is complete.

    Polygon data is written into the serialised dict only, not back onto G.graph,
    so the same graph object can be reused across multiple layout methods without
    accumulating stale data.

    Parameters
    ----------
    G : nx.Graph
        Graph with positions, cluster IDs, and any other node attributes set.
    path : str
        Output file path. Parent directories are created if they do not exist.
    polygons : dict[str, dict[int, str]] or None
        Mapping of cluster_key → {cluster_id → polygon_string}, as returned by
        compute_cluster_polygons(). Pass None to omit polygon data.
    strip_keys : list[str] or None
        Node attribute keys to exclude from the output. Defaults to ['vector'].
        Pass an empty list to keep all attributes.
    """
    if strip_keys is None:
        strip_keys = ['vector']

    data = nx.json_graph.node_link_data(G, edges='links')

    if polygons:
        for cluster_key, poly_dict in polygons.items():
            if not poly_dict:
                continue
            data['graph'][f'{cluster_key}_polygons'] = ';'.join(
                poly_dict.get(i, '') for i in range(max(poly_dict) + 1)
            )

    if strip_keys:
        for node_data in data['nodes']:
            for key in strip_keys:
                node_data.pop(key, None)

    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(data, f, indent=4, cls=_NumpyEncoder)