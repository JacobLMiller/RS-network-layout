import os
import pickle
import re

import networkx as nx
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer


def preprocess(
    G: nx.Graph,
    rich_type: str = 'rich',
    data_key: str = 'data',
    data_format: str = 'text',
    vector_key: str = 'vector',
    cache_dir: str = 'pickle',
    graphml_path: str | None = None,
    verbose: bool = False,
) -> nx.Graph:
    """
    Attach a vector embedding to each rich node of a bipartite graph.

    Rich nodes (those where G.nodes[n]['type'] == rich_type) must carry source data
    at data_key. After preprocessing every rich node will have a numpy float64 array
    at vector_key. Sparse nodes are not modified.

    Modifies G in place and returns it.

    Parameters
    ----------
    G : nx.Graph
        Bipartite NetworkX graph. Partition membership is read from the 'type' node attribute.
    rich_type : str
        Value of the 'type' attribute identifying rich nodes.
    data_key : str
        Node attribute holding the source data (text string, keyword string, or numeric array).
    data_format : {'text', 'keywords', 'vector'}
        Describes the content at data_key:
        - 'text'     : free-text string or list of strings (one per data dimension).
                       Each string is tokenised into sentences, encoded with a sentence
                       transformer, then averaged. Multi-dimension vectors are concatenated.
        - 'keywords' : semicolon-separated keyword string. Keywords are lemmatised, encoded
                       individually, then averaged.
        - 'vector'   : already a numeric array. Cast to numpy float64; no encoding performed.
    vector_key : str
        Node attribute key where the resulting numpy array will be written.
    cache_dir : str
        Directory used to cache the sentence-to-vector map (text format only). Speeds up
        repeated runs on the same graph. Per-graph, per-key caches are stored as
        {cache_dir}/{graph_name}_{data_key}_emb_map.pkl.
    graphml_path : str or None
        If provided, write the graph (with vectors serialised as space-separated float
        strings) to this path. The in-memory graph retains numpy arrays. Default None.
    verbose : bool
        Print progress during sentence encoding.

    Returns
    -------
    nx.Graph
        The same graph G, modified in place.
    """
    if data_format == 'vector':
        _attach_vectors(G, rich_type, data_key, vector_key)
    elif data_format == 'text':
        _embed_text(G, rich_type, data_key, vector_key, cache_dir, verbose)
    elif data_format == 'keywords':
        _embed_keywords(G, rich_type, data_key, vector_key, verbose)
    else:
        raise ValueError(
            f"data_format must be one of 'vector', 'text', 'keywords'. Got '{data_format}'."
        )

    if graphml_path is not None:
        _write_graphml(G, vector_key, graphml_path, verbose)

    return G


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _attach_vectors(G: nx.Graph, rich_type: str, data_key: str, vector_key: str) -> None:
    """Cast existing numeric data at data_key to a numpy float64 array."""
    for node, data in G.nodes(data=True):
        if data.get('type') != rich_type:
            continue
        if data_key not in data:
            raise KeyError(f"Rich node '{node}' is missing attribute '{data_key}'.")
        data[vector_key] = np.asarray(data[data_key], dtype=np.float64)


def _embed_text(
    G: nx.Graph,
    rich_type: str,
    data_key: str,
    vector_key: str,
    cache_dir: str,
    verbose: bool,
) -> None:
    """
    Sentence-encode the text at data_key for every rich node.

    data_key may hold a single string or a list of strings (one per data dimension).
    Each string is tokenised into sentences, which are encoded and averaged to form a
    per-dimension vector. Vectors across dimensions are concatenated.

    The sentence-to-vector map is cached at {cache_dir}/{gname}_{data_key}_emb_map.pkl.
    Only sentences absent from the cache are re-encoded, so the function is safe to call
    incrementally as the graph grows.
    """
    gname = G.graph.get('name', '')
    cache_file = (
        os.path.join(cache_dir, f"{gname}_{data_key}_emb_map.pkl") if gname else None
    )

    rich_nodes = {n: d for n, d in G.nodes(data=True) if d.get('type') == rich_type}
    if not rich_nodes:
        return

    # Parse each node's data into a list-of-sentence-lists (one list per dimension)
    node_sentence_dims: dict[object, list[list[str]]] = {}
    all_sentences: set[str] = set()
    for node, data in rich_nodes.items():
        if data_key not in data:
            raise KeyError(f"Rich node '{node}' is missing attribute '{data_key}'.")
        raw = data[data_key]
        if isinstance(raw, str):
            raw = [raw]
        dims = [sent_tokenize(text) for text in raw]
        node_sentence_dims[node] = dims
        for sents in dims:
            all_sentences.update(sents)

    # Load existing cache; encode only sentences that are not yet cached
    embed_map: dict[str, np.ndarray] = {}
    if cache_file and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            embed_map = pickle.load(f)

    missing = [s for s in all_sentences if s not in embed_map]
    if missing:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        new_vecs = model.encode(missing, show_progress_bar=verbose)
        embed_map.update(zip(missing, new_vecs))
        if cache_file:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(embed_map, f)

    # Build per-node vectors: average sentences per dimension, concatenate dimensions
    emb_dim = next(iter(embed_map.values())).shape[0]
    for node, dims in node_sentence_dims.items():
        vecs_per_dim = [
            np.mean([embed_map[s] for s in sents], axis=0) if sents else np.zeros(emb_dim)
            for sents in dims
        ]
        G.nodes[node][vector_key] = np.concatenate(vecs_per_dim)


def _embed_keywords(
    G: nx.Graph,
    rich_type: str,
    data_key: str,
    vector_key: str,
    verbose: bool,
) -> None:
    """
    Encode the semicolon-separated keywords at data_key for every rich node.

    Each keyword is lowercased, punctuation-stripped, and lemmatised. Unique keywords
    across all rich nodes are encoded in one batch, then averaged per node.
    """
    lemmatizer = WordNetLemmatizer()
    rich_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get('type') == rich_type]
    if not rich_nodes:
        return

    node_words: dict[object, list[str]] = {}
    all_words: set[str] = set()
    for node, data in rich_nodes:
        raw = data.get(data_key, '')
        words = []
        for word in raw.split('; '):
            word = word.lower().replace('-', ' ').replace('/', ' ').replace('_', ' ')
            word = re.sub(r'(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?', '', word)
            word = lemmatizer.lemmatize(word).strip()
            if word:
                words.append(word)
                all_words.add(word)
        node_words[node] = words

    model = SentenceTransformer('all-MiniLM-L6-v2')
    word_list = list(all_words)
    vecs = model.encode(word_list, show_progress_bar=verbose)
    embed_map = dict(zip(word_list, vecs))
    emb_dim = vecs.shape[1]

    for node, words in node_words.items():
        G.nodes[node][vector_key] = (
            np.mean([embed_map[w] for w in words], axis=0) if words else np.zeros(emb_dim)
        )


def _write_graphml(
    G: nx.Graph, vector_key: str, path: str, verbose: bool
) -> None:
    """
    Write G to a GraphML file. numpy vector arrays are serialised as space-separated
    float strings; the in-memory graph is not mutated.
    """
    H = G.copy()
    for _, data in H.nodes(data=True):
        if vector_key in data and isinstance(data[vector_key], np.ndarray):
            data[vector_key] = ' '.join(f'{x:.8g}' for x in data[vector_key])
    nx.write_graphml(H, path)
    if verbose:
        print(f"Wrote intermediate graph to '{path}'.")
