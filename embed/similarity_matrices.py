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


def embed_rich_by_abstract(G: nx.classes.graph.Graph, ret_emb: bool = True, abstract_key: str = 'data',
                            rich_type: str = 'rich', verbose=True):
    """
    Extract rich nodes and create a word embedding by encoding vector data with a sentence
    transformer, returning either a numpy array or a dict of node→vector pairs.

    Caches sentence-to-vector mapping in pickle/{G.name}_emb_map.pkl to prevent unnecessary
    recalculation. If the graph changes, delete that file to regenerate.

    Parameters
    ----------
    G : nx.classes.graph.Graph
        Graph with rich nodes ('type' == rich_type) whose data is stored in node[abstract_key].
    ret_emb : bool
        If True, returns embedding as a numpy array.
        If False, returns a dict of {node_id: vector}.
    abstract_key : str
        Key in node data to access the text content.
    rich_type : str
        Value of the 'type' attribute identifying rich nodes.
    """
    gname = G.graph['name'] if 'name' in G.graph else ""
    embed_map_file = f"pickle/{gname}_emb_map.pkl"
    _G = deepcopy(G)

    Rich = getType(_G, nodetype=rich_type)

    data_sentences = set()
    for node, node_data in Rich.nodes(data=True):
        rich_data = node_data[abstract_key]
        if isinstance(rich_data, str):
            rich_data = [rich_data]

        node_data['sentences'] = []
        for rich_data_i in rich_data:
            sentences_i = sent_tokenize(rich_data_i)
            node_data['sentences'].append(sentences_i)
            data_sentences.update(sentences_i)

    if os.path.exists(embed_map_file):
        with open(embed_map_file, 'rb') as fdata:
            embed_map = pickle.load(fdata)
    else:
        list_abstracts = list(data_sentences)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(list_abstracts, show_progress_bar=verbose)
        if verbose:
            print(f"Size of embeddings: {embeddings.shape}")

        embed_map = dict(zip(list_abstracts, embeddings))

        if not os.path.isdir("pickle"):
            os.mkdir("pickle")
        if gname:
            with open(embed_map_file, 'wb') as fdata:
                pickle.dump(embed_map, fdata)

    node_to_vec = dict()
    for (node, node_sents) in Rich.nodes(data='sentences'):
        vec_is = []
        for sentences_i in node_sents:
            # Average sentence vectors; all-MiniLM-L6-v2 encodes to 384 dimensions
            vec_i = sum(embed_map[w] for w in sentences_i) / len(sentences_i) if sentences_i else np.zeros(384)
            vec_is.append(vec_i)
        vec = np.concatenate(vec_is)
        node_to_vec[node] = vec

    if ret_emb:
        return np.array(list(node_to_vec.values()))
    else:
        return node_to_vec


def embed_rich_by_keywords(G: nx.classes.graph.Graph, ret_emb: bool = True,
                            keywords_key: str = 'seminar_keywords', rich_type: str = 'rich'):
    """
    Embed rich nodes by averaging sentence-transformer encodings of their keywords.

    Parameters
    ----------
    G : nx.classes.graph.Graph
        Graph with rich nodes ('type' == rich_type) whose keywords are in node[keywords_key].
    ret_emb : bool
        If True, returns embedding as a numpy array.
        If False, returns a dict of {node_id: vector}.
    keywords_key : str
        Key in node data to access the semicolon-separated keywords.
    rich_type : str
        Value of the 'type' attribute identifying rich nodes.
    """
    Rich = getType(G, nodetype=rich_type)

    words = set()
    lemmatizer = WordNetLemmatizer()
    for node, data in Rich.nodes(data=True):
        mywords = list()
        if keywords_key in data:
            for word in data[keywords_key].split("; "):
                word = word.lower()
                word = word.replace("-", " ").replace("/", " ").replace("_", " ")
                word = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", word)
                word = lemmatizer.lemmatize(word)

                if len(word) < 1:
                    continue
                words.add(word)
                mywords.append(word)

            data['words'] = mywords
        else:
            print(f"Node {node} does not have keywords")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    list_words = list(words)
    embeddings = model.encode(list_words, show_progress_bar=True)
    print(f"Size of keyword embeddings: {embeddings.shape}")

    embed_map = dict(zip(list_words, embeddings))

    if ret_emb:
        X = np.zeros((Rich.number_of_nodes(), embeddings.shape[1]))

        for i, (node, data) in enumerate(Rich.nodes(data=True)):
            # Average keyword vector; all-MiniLM-L6-v2 encodes to 384 dimensions
            myvec = sum(embed_map[w] for w in data['words']) / len(data['words']) if data['words'] else np.zeros(384)
            X[i] = myvec

        return X

    else:
        node_to_vec = dict()
        for node, data in Rich.nodes(data=True):
            myvec = sum(embed_map[w] for w in data['words']) / len(data['words']) if data['words'] else np.zeros(384)
            node_to_vec[node] = myvec
        return node_to_vec


def jaccard_coathorship_similarity(G: nx.classes.graph.Graph, ret_nodelist: bool = False,
                                   sparse_type: str = 'sparse') -> np.ndarray | tuple[np.ndarray, list[str]]:
    """
    Returns a similarity matrix between the sparse nodes of the graph using the Jaccard
    index of neighboring rich nodes.

    Caches matrix and node order in pickle/{G.name}_jac.pkl to prevent unnecessary
    recalculation. Delete the file if the graph has been replaced.

    Parameters
    ----------
    G : nx.classes.graph.Graph
        Bipartite graph with sparse nodes ('type' == sparse_type) and rich nodes.
    ret_nodelist : bool
        If False, returns the matrix only.
        If True, returns (matrix, nodelist) where nodelist is the node order of the matrix.
    sparse_type : str
        Value of the 'type' attribute identifying sparse nodes.
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
                pickle.dump((jac_matrix, sparse_nodes), fdata)

    if ret_nodelist:
        return jac_matrix, sparse_nodes
    else:
        return jac_matrix


def graph_theoretic_dist(G: nx.classes.graph.Graph, rich_type: str = 'rich',
                         sparse_type: str = 'sparse', node_order=None):
    if node_order is None:
        rich_and_sparse = list(getType(G, rich_type).nodes) + list(getType(G, sparse_type).nodes)
    else:
        rich_and_sparse = node_order

    dist = nx.floyd_warshall_numpy(G, nodelist=rich_and_sparse)
    is_inf = dist == np.inf

    num_rich = getType(G, rich_type).number_of_nodes()
    rich_to_sparse_dist = dist[:num_rich, num_rich:]

    is_inf = rich_to_sparse_dist == np.inf
    max_finite_dist = rich_to_sparse_dist[~is_inf].max()
    rich_to_sparse_dist[is_inf] = max_finite_dist

    return rich_to_sparse_dist
