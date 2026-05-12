import numpy as np
from .similarity_matrices import embed_rich_by_abstract, jaccard_coathorship_similarity, graph_theoretic_dist, embed_rich_by_keywords
from ._utils import dimension_reduction, getType
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from itertools import chain
import networkx as nx

class UnionGraph():
    """
    Create distance matrices for rich nodes, sparse nodes, and distances between them,
    then creates a unified distance matrix composed of those distance matrices normalized,
    and inputs it to a traditional dimension reduction algorithm (t-SNE, UMAP etc).

    Parameters
    ----------
    G :
        NetworkX bipartite graph of rich and sparse nodes.

    rich_type :
        'type' attribute value of rich nodes.

    sparse_type :
        'type' attribute value of sparse nodes.

    embed_rich_by : one of 'abstract', 'data', or 'keywords'
        Create sentence embedding on the abstract/data or keywords of the rich nodes.

    rich_embed_data_loc :
        Key name where the abstract/data or keywords are stored in rich node attributes.
    """
    def __init__(self, G: nx.classes.graph.Graph, rich_type: str = 'rich', sparse_type: str = 'sparse',
                 embed_rich_by: str = 'data', rich_embed_data_loc: str = 'data', verbose: int = 0):
        self.G = G
        self.rich_type = rich_type
        self.sparse_type = sparse_type
        self.embed_rich_by = embed_rich_by
        self.rich_embed_data_loc = rich_embed_data_loc
        self.verbose = verbose

    def _rich_sim(self, n_neighbors=15, mode='connectivity'):
        """
        Returns
        -------
        (rich_list, distance_matrix) tuple, where distance_matrix is the connectivity matrix
        of the n_neighbors nearest neighbors in embedding space, and rich_list is the node order.
        """
        if self.embed_rich_by == 'abstract' or self.embed_rich_by == 'data':
            rich_embed_method = embed_rich_by_abstract
        elif self.embed_rich_by == 'keywords':
            rich_embed_method = embed_rich_by_keywords
        else:
            raise ValueError(f"embed_rich_by should be one of 'abstract', 'data', or 'keywords'. "
                             f"Was '{self.embed_rich_by}' instead.")

        rich_emb_pos = rich_embed_method(self.G, False, self.rich_embed_data_loc, self.rich_type)
        rich_list = list(rich_emb_pos.keys())
        rich_nn = NearestNeighbors(n_neighbors=n_neighbors).fit(
            list(rich_emb_pos.values())
        ).kneighbors_graph(mode=mode).toarray()

        rich_nn = rich_nn + rich_nn.T
        rich_nn = MinMaxScaler().fit_transform(rich_nn)

        if self.verbose > 1:
            print("Created distance matrix of rich embedding.")

        return rich_list, rich_nn

    def fruchtermann(self, sparse_weight=1, rich_weight=1):
        """
        Keeps rich-sparse edges and introduces rich-rich and sparse-sparse edges too.
        Rich-rich edges connect the 15 nearest neighbors in embedding space.
        Sparse-sparse edges use Jaccard similarity of shared rich neighbors.
        An embedding is then generated with NetworkX's Fruchtermann-Reingold algorithm.
        """
        union_graph = self.get_union_graph(sparse_weight, rich_weight)
        pos = nx.spring_layout(union_graph)
        return pos

    def kamada_kawai(self, sparse_weight=1, rich_weight=1):
        """
        Same graph construction as fruchtermann(), but uses the Kamada-Kawai algorithm.
        """
        union_graph = self.get_union_graph(sparse_weight, rich_weight)
        return nx.kamada_kawai_layout(union_graph)

    def get_union_graph(self, sparse_weight=1, rich_weight=1, label_edges=False):
        rich_list, rich_nn = self._rich_sim(mode='connectivity')
        rich_nn *= rich_weight
        Rich = nx.from_numpy_array(rich_nn, nodelist=rich_list)
        if label_edges:
            for _, _, data in Rich.edges(data=True):
                data['relation'] = 'r2r'

        if self.verbose:
            print("Created rich graph")

        sparse_jac_emb, sparse_list = jaccard_coathorship_similarity(
            self.G, ret_nodelist=True, sparse_type=self.sparse_type
        )
        np.fill_diagonal(sparse_jac_emb, 0)
        sparse_jac_emb = MinMaxScaler().fit_transform(sparse_jac_emb)
        sparse_jac_emb *= sparse_weight
        Sparse = nx.from_numpy_array(sparse_jac_emb, nodelist=sparse_list)
        if label_edges:
            for _, _, data in Sparse.edges(data=True):
                data['relation'] = 's2s'

        if self.verbose:
            print("Created sparse graph")

        union_graph = nx.compose_all([self.G, Rich, Sparse])
        if label_edges:
            for _, _, data in union_graph.edges(data=True):
                if 'relation' not in data:
                    data['relation'] = 'r2s'

        if self.verbose:
            print("Created union graph")
        return union_graph

    def dim_reduction(self, alg='UMAP', pca_dim=50):
        """
        Create a distance measure between all nodes. Three types of distances are considered:
            1. Rich-rich distances (from sentence embedding).
            2. Rich-sparse distances (graph-theoretic shortest path).
            3. Sparse-sparse distances (1 - Jaccard similarity of shared rich neighbors).

        All three are separately scaled so their maximum equals 1, then merged into a single
        distance matrix and passed to a dimension reduction algorithm.
        """
        if self.embed_rich_by == 'abstract' or self.embed_rich_by == 'data':
            rich_embed_method = embed_rich_by_abstract
        elif self.embed_rich_by == 'keywords':
            rich_embed_method = embed_rich_by_keywords
        else:
            raise ValueError(f"embed_rich_by should be one of 'abstract', 'data', or 'keywords'. "
                             f"Got '{self.embed_rich_by}' instead.")

        rich_dict = rich_embed_method(self.G, False, self.rich_embed_data_loc, self.rich_type)
        rich_nodes = list(rich_dict.keys())
        rich_vec = np.array(list(rich_dict.values()))
        rich_vec = PCA(n_components=pca_dim).fit_transform(rich_vec)
        r2r_dist = pairwise_distances(rich_vec)

        if self.verbose:
            print("Distance matrix: Created rich-rich block.")

        jac_sim_mat, sparse_nodes = jaccard_coathorship_similarity(
            self.G, ret_nodelist=True, sparse_type=self.sparse_type
        )
        s2s_dist = 1 - jac_sim_mat

        if self.verbose:
            print("Distance matrix: Created sparse-sparse block.")

        node_order = rich_nodes + sparse_nodes
        r2s_dist = graph_theoretic_dist(
            self.G, rich_type=self.rich_type, sparse_type=self.sparse_type, node_order=node_order
        )

        if self.verbose:
            print("Distance matrix: Created rich-sparse block.")

        r2r_dist, s2s_dist, r2s_dist = [MinMaxScaler().fit_transform(X)
                                         for X in [r2r_dist, s2s_dist, r2s_dist]]

        # Assemble full distance matrix:
        # [ r2r  r2s ]
        # [ r2s' s2s ]
        AB = np.concatenate((r2r_dist, r2s_dist), axis=1)
        CD = np.concatenate((r2s_dist.T, s2s_dist), axis=1)
        full_dist = np.concatenate((AB, CD), axis=0)

        if self.verbose:
            print("Distance matrix created. Running dimension reduction.")

        Y = dimension_reduction(full_dist, alg, X_is_dist=True)

        pos = {node: vec for node, vec in zip(node_order, Y)}
        return pos
