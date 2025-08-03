import networkx as nx
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import re
import os
import pickle
from copy import deepcopy

from ._utils import getType


def embed_papers_by_abstract(G: nx.classes.graph.Graph, ret_emb: bool= True, abstract_key: str='data', paper_type: str='rich', verbose=True):
    """
    Extract Paper node and create word embedding by training sentence transformer on abstract and reduce the embedding to 2 dimensions.

    Caches sentence-to-vector mapping in pickle/{G.name}_emb_map.pkl to prevent unnecessary recalculation. If there is an error upon updating
    the graph, try deleting the file as it is most probably outdated.

    Parameters
    ----------

    G : nx.classes.graph.Graph
        Graph with Paper nodes ('type' == paper_type) which have abstracts stored in data[abstract_key]

    ret_emb : bool
        If True, returns sentence embedding as Numpy array
        If False, returns dictionary of node:vector key-value pairs

    abstract_key : str
        Key in node data to access abstracts
    
    paper_type : str
        Name of type of paper nodes
    
    Returns
    -------
      Sentence embedding as a np.ndarray if ret_emb is True, else returns embedding as a dictionary
      with the node and its corresponding vector as key-value pairs.
    """
    gname = G.graph['name'] if 'name' in G.graph else ""
    embed_map_file = f"pickle/{gname}_emb_map.pkl"
    _G = deepcopy(G)
    
    Papers = getType(_G, nodetype=paper_type)

    # Get abstracts from Papers
    abstracts = set()
    for node, node_data in Papers.nodes(data=True):
        abstract = node_data[abstract_key]
        if not isinstance(abstract, list):
            abstract = [abstract]
            
        sentences = sent_tokenize(node_data[abstract_key])
        node_data['sentences'] = sentences
        abstracts.update(sentences)

    if os.path.exists(embed_map_file): 
        with open(embed_map_file, 'rb') as fdata:
            embed_map = pickle.load(fdata)
    else:
        list_abstracts = list(abstracts)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(list_abstracts,show_progress_bar=verbose)
        if verbose: print(f"Size of embeddings: {embeddings.shape}")

        embed_map = dict(zip(list_abstracts,embeddings))

        if not os.path.isdir("pickle"): 
            os.mkdir("pickle")
        if gname: 
            with open(embed_map_file, 'wb') as fdata:
                pickle.dump(embed_map,fdata)



    if ret_emb:
        X = np.zeros( (Papers.number_of_nodes(), 384) )

        for i,(node,node_data) in enumerate(Papers.nodes(data=True)):
            myvec = sum(embed_map[w] for w in node_data['sentences']) / len(node_data['sentences']) if node_data['sentences'] else np.zeros(384) # average sentence vector, all-MiniLM-L6-v2 transformer encodes to 384 dimensiosn
            X[i] = myvec

        return X
    
    else:
        node_to_vec = dict()
        for node,node_data in Papers.nodes(data=True):
            myvec = sum(embed_map[w] for w in node_data['sentences']) / len(node_data['sentences']) if node_data['sentences'] else np.zeros(384) # average sentence vector
            node_to_vec[node] = myvec
        return node_to_vec

def embed_papers_by_keywords(G: nx.classes.graph.Graph, ret_emb: bool= True, keywords_key: str='seminar_keywords', paper_type: str='Paper'):
    Papers = getType(G, nodetype=paper_type)

    # Get keywords from Papers
    words = set()
    lemmatizer = WordNetLemmatizer()
    for node, data in G.nodes(data=True):
        mywords = list()
        if keywords_key in data:
            for word in data[keywords_key].split("; "):
                word = word.lower()
                word = word.replace("-", " ").replace("/", " ").replace("_", " ")
                word = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", word)
                word = lemmatizer.lemmatize(word)
               
                if len(word) < 1: continue
                words.add(word)
                mywords.append(word)
            
            data['words'] = mywords
        else:
            print(f"Node {node} does not have keywords")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    list_words = list(words)
    embeddings = model.encode(list_words,show_progress_bar=True)
    print(f"Size of keyword embeddings: {embeddings.shape}")

    embed_map = dict(zip(list_words,embeddings))

    if ret_emb:
        X = np.zeros( (Papers.number_of_nodes(), embeddings.shape[1]) )

        for i,(node,data) in enumerate(Papers.nodes(data=True)):
            # Average keyword vector
            # all-MiniLM-L6-v2 transformer encodes to 384 dimensions
            myvec = sum(embed_map[w] for w in data['words']) / len(data['words']) if data['words'] else np.zeros(384)
            X[i] = myvec

        return X
    
    else:
        node_to_vec = dict()
        for node,data in Papers.nodes(data=True):
            myvec = sum(embed_map[w] for w in data['words']) / len(data['words']) if data['words'] else np.zeros(384) # average sentence vector
            node_to_vec[node] = myvec
        return node_to_vec


def jaccard_coathorship_similarity(G: nx.classes.graph.Graph, ret_nodelist:bool=False, sparse_type:str='sparse') -> np.ndarray | tuple[np.ndarray, list[str]]:
    """
    Returns a similariy matrix between the sparse nodes of the graph using the jaccard index of neighboring rich nodes

    Caches matrix and node order in pickle/{G.name}_jac.pkl to prevent unnecessary recalculation. If a previously calculated graph
    was replaced by a new graph of the same name, this pickle file should be deleted as it is outdated.
    
    Parameters
    ----------
    
    G : nx.classes.graph.Graph
        Bipartite graph with sparse nodes ('type' == sparsetype) and rich nodes

    ret_nodelist : bool
        If False, simply returns matrix
        If True, returns tuple (mat, nodelist) where mat is the similarity matrix and nodelist is node order of mat

    sparse_type : str
        Name of type of sparse nodes (i.e. G.nodes[node]['type'] == sparse_type)

    """
    from sklearn.metrics import pairwise_distances

    gname = G.graph['name'] if 'name' in G.graph else ""
    jac_pkl_file = f"pickle/{gname}_jac.pkl"

    if os.path.exists(jac_pkl_file): 
        with open(jac_pkl_file, 'rb') as fdata:
            jac_matrix, sparse_nodes = pickle.load(fdata)
    else:
        node_list = list(G.nodes())
        node_index = {node: i for i, node in enumerate(node_list)}

        sparse_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == sparse_type]
        sparse_indices = [node_index[n] for n in sparse_nodes]
        rich_indices = [i for i in range(len(node_list)) if i not in sparse_indices]

        A = nx.to_numpy_array(G, nodelist=node_list, dtype=bool)
        sub_A = A[sparse_indices][:, rich_indices]

        jac_matrix = 1 - pairwise_distances(sub_A, metric='jaccard')

        if not os.path.isdir("pickle"): 
            os.mkdir("pickle")
        if gname: 
            with open(jac_pkl_file, 'wb') as fdata:
                pickle.dump((jac_matrix, sparse_nodes),fdata)

    if ret_nodelist:
        return jac_matrix, sparse_nodes
    else:
        return jac_matrix

def graph_theoretic_dist(G: nx.classes.graph.Graph, paper_type='Paper', author_type='Author', node_order=None):
    
    if node_order is None:
        papers_and_authors = list(getType(G, paper_type).nodes) + list(getType(G, author_type).nodes)
    else:
        papers_and_authors = node_order # Assumes first paper nodes, then author nodes
    dist = nx.floyd_warshall_numpy(G, nodelist=papers_and_authors)
    is_inf = dist == np.inf

    # Only paper to author distances
    num_papers = getType(G, paper_type).number_of_nodes()
    paper_to_auth_dist = dist[:num_papers, num_papers:]
    
    is_inf = paper_to_auth_dist == np.inf
    max_finite_dist = paper_to_auth_dist[~is_inf].max()
    paper_to_auth_dist[is_inf] = max_finite_dist

    return paper_to_auth_dist

