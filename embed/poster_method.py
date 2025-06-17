import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from sklearn.decomposition import PCA, TruncatedSVD
from .similarity_matrices import embed_papers_by_keywords, embed_papers_by_abstract
from ._utils import dimension_reduction
from scipy.sparse import coo_matrix



class Poster():
    def __init__(self, G: nx.classes.graph.Graph, paper_type:str='rich', author_type:str='sparse', embed_papers_by:str='data', paper_embed_data_loc='data'):
        self.G = G
        self.paper_type = paper_type
        self.author_type = author_type
        self.embed_papers_by = embed_papers_by
        self.paper_embed_data_loc = paper_embed_data_loc

    def get_author_influence(self,dims=50):

        unodes = list()
        vnodes = list()
        for u,v in self.G.edges():
            if self.G.nodes[u]["type"] == self.author_type: u,v = v,u

            unodes.append(int(self.G.nodes[u]['id'].replace("r_", "")))
            vnodes.append(int(self.G.nodes[v]['id'].replace("s_", "")))
        
        sparsemat = coo_matrix((np.ones(len(unodes)), (unodes, vnodes)))

        author_influence = TruncatedSVD(n_components=dims).fit_transform(sparsemat)

        return author_influence

    def layout(self, pca_dim=50):
        """
        The Paper nodes are embedded by creating a sentence embedding of the Paper nodes (trained on either their keywords or on the abstract),
        and creating a 2D projection using UMAP. The Author nodes are placed in the geometric center of the Paper nodes that they share an
        edge with.
        """
        paper_emb_pos = self._paper_emb()
        high_dim_emb = np.array(list(paper_emb_pos.values()))
        word_influence = PCA(n_components=pca_dim).fit_transform(high_dim_emb)

        author_influence = self.get_author_influence()

        # X = np.hstack((word_influence,author_influence))
        X = word_influence
        print(X)
        print(X.shape)

        paper_Y = dimension_reduction(X, 'UMAP', False)
        paper_pos = dict(zip(paper_emb_pos.keys(), paper_Y))
        
        author_nodes = [node for node, nodetype in self.G.nodes(data='type') if nodetype == self.author_type]
        author_pos = {
            node : np.mean([paper_pos[paper_node] for paper_node in self.G[node]], 0) for node in author_nodes
        }

        return paper_pos | author_pos

        

    def _paper_emb(self):
        """

        """
        if self.embed_papers_by == 'abstract' or self.embed_papers_by == "data":
            paper_embed_method = embed_papers_by_abstract
        elif self.embed_papers_by == 'keywords':
            paper_embed_method = embed_papers_by_keywords
        else:
            raise ValueError(f"embed_paper_by should be one of 'abstract' or 'keywords'. Was {self.embed_papers_by} instead.")
        
        paper_emb_pos = paper_embed_method(self.G, False, self.paper_embed_data_loc, self.paper_type)
        
        return paper_emb_pos