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
    Extract Paper node and create word embedding by training sentence transformer on abstract and reduce the embedding to 2 dimensions

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
    _G = deepcopy(G)
    
    Papers = getType(_G, nodetype=paper_type)

    # Get abstracts from Papers
    abstracts = set()
    for node, data in Papers.nodes(data=True):
        sentences = sent_tokenize(data[abstract_key])
        data['sentences'] = sentences
        abstracts.update(sentences)

    if os.path.exists(f"pickle/{gname}.pkl"): 
        with open(f"pickle/{gname}.pkl", 'rb') as fdata:
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
            with open(f"pickle/{gname}.pkl", 'wb') as fdata:
                pickle.dump(embed_map,fdata)



    if ret_emb:
        X = np.zeros( (Papers.number_of_nodes(), 384) )

        for i,(node,data) in enumerate(Papers.nodes(data=True)):
            myvec = sum(embed_map[w] for w in data['sentences']) / len(data['sentences']) if data['sentences'] else np.zeros(384) # average sentence vector, all-MiniLM-L6-v2 transformer encodes to 384 dimensiosn
            X[i] = myvec

        return X
    
    else:
        node_to_vec = dict()
        for node,data in Papers.nodes(data=True):
            myvec = sum(embed_map[w] for w in data['sentences']) / len(data['sentences']) if data['sentences'] else np.zeros(384) # average sentence vector
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


def jaccard_coathorship_similarity(G: nx.classes.graph.Graph, ret_nodelist:bool=False, author_type:str='Author') -> np.ndarray | tuple[np.ndarray, list[str]]:
    """
    Returns a similariy matrix between the authors of the graph using the jaccard index of co-authored papers
    
    Parameters
    ----------
    
    G : nx.classes.graph.Graph
        Bipartite graph with Author nodes ('type' == 'Author') and Paper nodes ('type' == 'Paper')

    ret_nodelist : bool
        If False, simply returns matrix
        If True, returns tuple (mat, nodelist) where mat is the similarity matrix and nodelist is node order of mat

    author_type : str
        Name of type of author nodes (i.e. G.nodes[node]['type'] == author_type)

    """
    authors = getType(G, nodetype=author_type).nodes()

    jac_matrix = list()

    for node1 in authors:
        row = list()
        for node2 in authors:
            nbrs1 = set(G[node1])
            nbrs2 = set(G[node2])
            jaccard = len(nbrs1 & nbrs2) / len(nbrs1 | nbrs2)
            row.append(jaccard)
        jac_matrix.append(row)

    np_matrix = np.array(jac_matrix)
    if ret_nodelist:
        return np_matrix, list(authors)
    else:
        return np_matrix

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
    

    # is_author = np.array([data['type'] == 'Author' for _, data in G.nodes(data=True)])
    # full_false = np.full(is_author, False)
    # is_paper_to_author = np.array([is_author if data['type'] == 'Paper' else full_false for _, data in G.nodes(data=True)])

    # paper_to_auth_simi = dist[is_paper_to_author].reshape(getPapers(G).number_of_nodes(), getAuthors(G).number_of_nodes())
    
    # is_inf = paper_to_auth_simi == np.inf
    # max_finit_dist = paper_to_auth_simi[~is_inf].max()
    # paper_to_auth_simi[~is_inf] = max_finit_dist - paper_to_auth_simi[~is_inf]
    # paper_to_auth_simi[is_inf] = 0


    # return paper_to_auth_simi

