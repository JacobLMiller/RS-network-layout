import networkx as nx
from .similarity_matrices import embed_rich_by_abstract, embed_rich_by_keywords, jaccard_coathorship_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import orthogonal_procrustes
import numpy as np

def procrustes_layout(
        G: nx.classes.graph.Graph,
        rich_type: str = 'rich',
        sparse_type: str = 'sparse',
        embed_rich_by: str = 'data',
        rich_embed_data_loc: str = 'data',
        transform_author: bool = True,
):
    """
    Rich and sparse nodes are first embedded independently of their relationship with each other.
    For rich nodes, a sentence embedding is generated from the node data, and a nearest-neighbor
    graph is laid out with Fruchtermann-Reingold. Sparse nodes are embedded via Jaccard similarity
    of shared rich neighbors.

    The two embeddings are then aligned using Procrustes analysis: one node type is placed at the
    geometric centroid of its neighbors in the other embedding (the pseudo-embedding), and the
    optimal rotation/scale is found. That transformation is applied to the independent embedding.

    Parameters
    ----------
    G : NetworkX Graph
        Bipartite graph with rich and sparse nodes.
    rich_type : str
        'type' attribute value of rich nodes.
    sparse_type : str
        'type' attribute value of sparse nodes.
    embed_rich_by : str
        One of 'abstract', 'data', or 'keywords'.
    rich_embed_data_loc : str
        Node attribute key where text data is stored.
    transform_author : bool
        If True, rich embedding is fixed and sparse is transformed to align.
        If False, roles are swapped.
    """
    if embed_rich_by == 'abstract' or embed_rich_by == 'data':
        rich_embed_method = embed_rich_by_abstract
    elif embed_rich_by == 'keywords':
        rich_embed_method = embed_rich_by_keywords
    else:
        raise ValueError(f"embed_rich_by should be one of 'abstract', 'data', or 'keywords'. "
                         f"Was '{embed_rich_by}' instead.")

    rich_emb_pos = rich_embed_method(G, False, rich_embed_data_loc, rich_type)
    rich_list = list(rich_emb_pos.keys())
    nn = NearestNeighbors(n_neighbors=15).fit(list(rich_emb_pos.values())).kneighbors_graph().toarray()
    Rich = nx.from_numpy_array(nn, nodelist=rich_list)
    rich_pos = nx.spring_layout(Rich)

    sparse_jac_emb, sparse_list = jaccard_coathorship_similarity(G, ret_nodelist=True, sparse_type=sparse_type)
    Sparse = nx.from_numpy_array(sparse_jac_emb, nodelist=sparse_list)
    sparse_pos = nx.spring_layout(Sparse)

    pos1, pos2 = (rich_pos, sparse_pos) if transform_author else (sparse_pos, rich_pos)
    emb1 = np.array(list(pos1.values()))
    emb2 = np.array(list(pos2.values()))
    emb2_gm = np.array([np.mean([pos2[nbr] for nbr in G[node]], 0) for node in pos1.keys()])

    mean1 = np.mean(emb1, 0)
    mean2_gm = np.mean(emb2_gm, 0)
    emb1 -= mean1
    emb2 -= mean2_gm
    emb2_gm -= mean2_gm

    norm1 = np.linalg.norm(emb1)
    norm2_gm = np.linalg.norm(emb2_gm)
    emb1 /= norm1
    emb2 /= norm2_gm
    emb2_gm /= norm2_gm

    emb2_gm_flipped = -emb2_gm
    R_flip, s_flip = orthogonal_procrustes(emb1, emb2_gm_flipped)
    emb2_gm_flipped_transformed = (emb2_gm_flipped @ R_flip.T) * s_flip
    mse_flip = np.mean(np.linalg.norm(emb2_gm_flipped_transformed - emb1, axis=1))

    R, s = orthogonal_procrustes(emb1, emb2_gm)
    emb2_gm_transformed = (emb2_gm @ R.T) * s
    mse_no_flip = np.mean(np.linalg.norm(emb2_gm_transformed - emb1, axis=1))

    if mse_flip < mse_no_flip:
        emb2 = ((-emb2) @ R_flip.T) * s_flip
    else:
        emb2 = (emb2 @ R.T) * s

    pos1 = {key: value for key, value in zip(pos1.keys(), emb1)}
    pos2 = {key: value for key, value in zip(pos2.keys(), emb2)}

    return pos1 | pos2


def box_procrustes_layout(
    G: nx.classes.graph.Graph,
    rich_type: str = 'rich',
    sparse_type: str = 'sparse',
    embed_rich_by: str = 'data',
    rich_embed_data_loc: str = 'data',
    transform_author: bool = True,
    verbose: int = 0,
):
    if verbose > 5:
        print(f"Running box procrustes layout with transform_author={transform_author}")

    if embed_rich_by == 'abstract' or embed_rich_by == 'data':
        rich_embed_method = embed_rich_by_abstract
    elif embed_rich_by == 'keywords':
        rich_embed_method = embed_rich_by_keywords
    else:
        raise ValueError(f"embed_rich_by should be one of 'abstract', 'data', or 'keywords'. "
                         f"Was '{embed_rich_by}' instead.")

    rich_emb_pos = rich_embed_method(G, False, rich_embed_data_loc, rich_type)
    rich_list = list(rich_emb_pos.keys())
    nn = NearestNeighbors(n_neighbors=15).fit(list(rich_emb_pos.values())).kneighbors_graph().toarray()
    Rich = nx.from_numpy_array(nn, nodelist=rich_list)
    rich_pos = nx.spring_layout(Rich)

    if verbose:
        print("Created rich part.")

    sparse_jac_emb, sparse_list = jaccard_coathorship_similarity(G, ret_nodelist=True, sparse_type=sparse_type)
    Sparse = nx.from_numpy_array(sparse_jac_emb, nodelist=sparse_list)
    sparse_pos = nx.spring_layout(Sparse)

    if verbose:
        print("Created sparse part.")

    pos1, pos2 = (rich_pos, sparse_pos) if transform_author else (sparse_pos, rich_pos)
    emb1 = np.array(list(pos1.values()))
    emb2 = np.array(list(pos2.values()))
    emb2_gm = np.array([np.mean([pos2[nbr] for nbr in G[node]], 0) for node in pos1.keys()])

    mean1 = np.mean(emb1, 0)
    mean2 = np.mean(emb2, 0)
    emb1 -= mean1
    emb2 -= mean2
    emb2_gm -= mean2

    max_norm1 = np.max(np.linalg.norm(emb1, axis=1))
    max_norm2 = np.max(np.linalg.norm(emb2, axis=1))
    emb1 /= max_norm1
    emb2 /= max_norm2
    emb2_gm /= max_norm2

    emb2_gm_flipped = -emb2_gm
    R_flip, s_flip = orthogonal_procrustes(emb1, emb2_gm_flipped)
    emb2_gm_flipped_transformed = (emb2_gm_flipped @ R_flip.T) * s_flip
    mse_flip = np.mean(np.linalg.norm(emb2_gm_flipped_transformed - emb1, axis=1))

    R, s = orthogonal_procrustes(emb1, emb2_gm)
    emb2_gm_transformed = (emb2_gm @ R.T) * s
    mse_no_flip = np.mean(np.linalg.norm(emb2_gm_transformed - emb1, axis=1))

    # Don't multiply by s to preserve scale
    if mse_flip < mse_no_flip:
        emb2 = ((-emb2) @ R_flip.T)
    else:
        emb2 = (emb2 @ R.T)

    pos1 = {key: value for key, value in zip(pos1.keys(), emb1)}
    pos2 = {key: value for key, value in zip(pos2.keys(), emb2)}

    return pos1 | pos2
