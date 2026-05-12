import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import eigh
import networkx as nx

from ._utils import dimension_reduction, place_sparse_nodes


class Semantic():
    def __init__(self, G: nx.Graph, rich_type: str = 'rich', sparse_type: str = 'sparse',
                 vector_key: str = 'vector', verbose: int = 0):
        """
        Parameters
        ----------
        G : nx.Graph
            Bipartite graph whose rich nodes already carry a vector embedding at vector_key,
            as produced by embed.preprocess().
        rich_type : str
            Value of the 'type' node attribute identifying rich nodes.
        sparse_type : str
            Value of the 'type' node attribute identifying sparse nodes.
        vector_key : str
            Node attribute key holding the pre-computed vector (numpy array or
            space-separated float string if loaded from GraphML).
        verbose : int
            Verbosity level.
        """
        self.G = G
        self.rich_type = rich_type
        self.sparse_type = sparse_type
        self.vector_key = vector_key
        self.verbose = verbose

    def layout(self, pca_dim: int = 50, laplacian_dims: int = 50, n_neighbors: int = 15) -> dict:
        """
        Compute a 2-D layout for all nodes.

        Rich nodes are positioned by running UMAP on a concatenation of:
          1. The pre-computed node vectors reduced to pca_dim dimensions with PCA.
          2. A spectral embedding of rich nodes derived from the normalised graph Laplacian
             (bipartite projection onto rich nodes), truncated to laplacian_dims dimensions.

        Sparse nodes are placed at the geometric centroid of their rich neighbours.

        Returns
        -------
        dict
            Mapping of node id → 2-D numpy array.
        """
        rich_vecs = self._read_vectors()
        rich_nodes = list(rich_vecs.keys())
        high_dim_emb = np.array(list(rich_vecs.values()))

        semantic_emb = PCA(n_components=pca_dim).fit_transform(high_dim_emb)

        if self.verbose:
            print(f"Semantic embedding: {semantic_emb.shape}")

        laplacian_emb = self._laplacian_embedding(rich_nodes, dims=laplacian_dims)

        # Pad if the graph is too small to yield laplacian_dims eigenvectors
        if laplacian_emb.shape[1] < laplacian_dims:
            pad = np.zeros((laplacian_emb.shape[0], laplacian_dims - laplacian_emb.shape[1]))
            laplacian_emb = np.hstack((laplacian_emb, pad))

        X = np.hstack((semantic_emb, laplacian_emb))

        if self.verbose:
            print(f"Combined embedding (semantic + Laplacian): {X.shape}")

        rich_Y = dimension_reduction(X, 'UMAP', False, n_neighbors=n_neighbors)
        rich_pos = dict(zip(rich_nodes, rich_Y))

        if self.verbose:
            print("Rich nodes positioned.")

        sparse_pos = place_sparse_nodes(self.G, rich_pos, sparse_type=self.sparse_type)

        if self.verbose:
            print("Sparse nodes positioned.")

        return rich_pos | sparse_pos

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_vectors(self) -> dict:
        """
        Return {node_id: np.ndarray} for all rich nodes.

        Accepts both numpy arrays (in-memory preprocessing) and space-separated float
        strings (vectors restored from a GraphML file).
        """
        result = {}
        for node, data in self.G.nodes(data=True):
            if data.get('type') != self.rich_type:
                continue
            if self.vector_key not in data:
                raise KeyError(
                    f"Rich node '{node}' has no '{self.vector_key}' attribute. "
                    "Call embed.preprocess() before running the layout."
                )
            raw = data[self.vector_key]
            if isinstance(raw, str):
                raw = [float(x) for x in raw.split()]
            result[node] = np.asarray(raw, dtype=float)
        return result

    def _laplacian_embedding(self, rich_nodes: list, dims: int = 50) -> np.ndarray:
        """
        Spectral embedding of rich nodes via the normalised Laplacian of the bipartite
        graph projected onto rich nodes (B @ B.T, where B is the biadjacency matrix).

        Returns an array of shape (len(rich_nodes), min(dims, n_rich - 1)).
        """
        sparse_nodes = [n for n, t in self.G.nodes(data='type') if t == self.sparse_type]

        B = nx.bipartite.biadjacency_matrix(
            self.G, row_order=rich_nodes, column_order=sparse_nodes
        ).toarray().astype(float)

        S = B @ B.T  # rich-to-rich co-occurrence

        degrees = np.maximum(S.sum(axis=1), 1e-10)
        d_inv_sqrt = 1.0 / np.sqrt(degrees)
        L_norm = np.eye(len(rich_nodes)) - (d_inv_sqrt[:, None] * S * d_inv_sqrt[None, :])

        safe_dims = min(dims, len(rich_nodes) - 1)
        _, eigvecs = eigh(L_norm, subset_by_index=[1, safe_dims])
        return eigvecs
