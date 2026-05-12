import os
import re

import numpy as np
import networkx as nx
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score

if os.name == "nt":
    os.environ["OMP_NUM_THREADS"] = "1"
    import warnings
    warnings.filterwarnings("ignore")


def assign_clusters(
    G: nx.Graph,
    rich_type: str = 'rich',
    k_coarse: int | None = None,
    k_fine: int | None = None,
    k_max_coarse: int = 8,
    k_max_fine: int = 7,
    coarse_key: str = 'c2',
    fine_key: str = 'c1',
    data_key: str = 'data',
    n_title_words: int = 1,
    interactive: bool = True,
    n_label_candidates: int = 10,
    keywords_key: str | None = 'seminar_keywords',
    keywords_sep: str = '; ',
    coarse_genre_key: str | None = None,
    fine_rep_title_key: str | None = None,
    vector_key: str = 'vector',
) -> nx.Graph:
    """
    Assign two nested levels of cluster labels to each rich node based on 2-D position,
    and derive a short title for each cluster via TF-IDF on the nodes' text descriptions.

    Coarse clusters (coarse_key) are computed first across all rich nodes using Ward
    agglomerative clustering. Fine clusters (fine_key) are then computed independently
    within each coarse group; their IDs are offset so they are globally unique across
    the whole graph.

    When k is not specified, the number of clusters at each level is chosen to maximise
    the silhouette score.

    Cluster titles are stored on the graph as semicolon-separated strings at
    G.graph['{coarse_key}_titles'] and G.graph['{fine_key}_titles'], ordered by cluster ID.

    Parameters
    ----------
    G : nx.Graph
        Graph with 2-D positions ('x', 'y') on rich nodes, as produced by a layout method.
    rich_type : str
        Value of the 'type' node attribute identifying rich nodes.
    k_coarse : int or None
        Fixed number of coarse clusters. None selects k automatically via silhouette score.
    k_fine : int or None
        Fixed number of fine clusters per coarse group. None selects k automatically.
    k_max_coarse : int
        Upper bound on the silhouette search range for the coarse level.
    k_max_fine : int
        Upper bound on the silhouette search range within each coarse group.
    coarse_key : str
        Node attribute key where coarse cluster IDs (integers) are written.
    fine_key : str
        Node attribute key where fine cluster IDs (integers) are written.
    data_key : str
        Node attribute key holding the text description used for title generation.
        If the value is a list of strings, they are joined with a space.
    n_title_words : int
        Number of top TF-IDF keywords to include in each cluster title.
    interactive : bool
        If True, print the top n_label_candidates TF-IDF label suggestions for each
        cluster and prompt the user to pick one (or type a custom label) in the terminal.
    n_label_candidates : int
        Number of candidate labels shown per cluster when interactive is True.
    keywords_key : str or None
        Node attribute holding a delimited keyword string (e.g. 'seminar_keywords').
        When set, cluster labels are derived from the most frequent keywords across nodes
        in each cluster rather than TF-IDF on free text. TF-IDF is used as a fallback for
        clusters where no keywords are present. Set to None to disable.
    keywords_sep : str
        Delimiter used to split the keywords_key string. Default '; ' (Dagstuhl style);
        use ', ' for Netflix-style comma-separated keywords.
    coarse_genre_key : str or None
        Node attribute holding a ', '-delimited genre string (e.g. 'genres').
        When set, coarse cluster labels are derived from genre frequency instead of TF-IDF
        or keywords. Fine cluster labels are unaffected.
    fine_rep_title_key : str or None
        Node attribute holding a human-readable title (e.g. 'title' for movies).
        When set, fine cluster labels are the title of the node whose embedding vector
        (at vector_key) is closest to the cluster centroid, giving a concrete anchor
        like "Cluster: The Dark Knight". TF-IDF / keyword labels are used as fallback
        when no node in a cluster has both a vector and a title.
    vector_key : str
        Node attribute holding the embedding vector used for centroid computation.
        Only relevant when fine_rep_title_key is set.

    Returns
    -------
    nx.Graph
        The same graph G, with coarse_key and fine_key written to each rich node and
        cluster title strings written to G.graph.
    """
    rich_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == rich_type]
    if not rich_nodes:
        return G

    X = np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in rich_nodes])

    # --- Coarse level ---
    if k_coarse is not None:
        coarse_labels = AgglomerativeClustering(
            n_clusters=min(k_coarse, len(rich_nodes) - 1), linkage='ward'
        ).fit_predict(X)
    else:
        coarse_labels = _best_ward_k(X, k_min=2, k_max=k_max_coarse)

    # --- Fine level: one clustering per coarse group, globally unique IDs ---
    fine_labels = np.full(len(rich_nodes), -1, dtype=int)
    offset = 0
    for c in np.unique(coarse_labels):
        mask = np.where(coarse_labels == c)[0]
        X_sub = X[mask]
        if len(X_sub) < 2:
            fine_labels[mask] = offset
            offset += 1
            continue
        if k_fine is not None:
            sub_labels = AgglomerativeClustering(
                n_clusters=min(k_fine, len(X_sub) - 1), linkage='ward'
            ).fit_predict(X_sub)
        else:
            sub_labels = _best_ward_k(X_sub, k_min=2, k_max=k_max_fine)
        fine_labels[mask] = sub_labels + offset
        offset += int(sub_labels.max()) + 1

    for node, c, f in zip(rich_nodes, coarse_labels, fine_labels):
        G.nodes[node][coarse_key] = int(c)
        G.nodes[node][fine_key] = int(f)

    # --- Collect per-cluster text docs and node lists ---
    coarse_docs: dict[int, list[str]] = {}
    fine_docs: dict[int, list[str]] = {}
    coarse_nodes_map: dict[int, list] = {}
    fine_nodes_map: dict[int, list] = {}
    for node, c, f in zip(rich_nodes, coarse_labels, fine_labels):
        raw = G.nodes[node].get(data_key, '')
        text = ' '.join(raw) if isinstance(raw, list) else str(raw)
        text = re.sub(r'<[^>]+>', ' ', text)
        coarse_docs.setdefault(int(c), []).append(text)
        fine_docs.setdefault(int(f), []).append(text)
        coarse_nodes_map.setdefault(int(c), []).append(node)
        fine_nodes_map.setdefault(int(f), []).append(node)

    coarse_texts = {c: ' '.join(texts) for c, texts in coarse_docs.items()}
    fine_texts = {f: ' '.join(texts) for f, texts in fine_docs.items()}

    # --- Build label candidates ---
    # Coarse: genre frequency if coarse_genre_key is set, else keyword/TF-IDF
    if coarse_genre_key:
        coarse_cands = _genre_candidates(coarse_nodes_map, G, coarse_genre_key, n_label_candidates)
    elif keywords_key:
        coarse_kw = _keyword_candidates(coarse_nodes_map, G, keywords_key, n_label_candidates, keywords_sep)
        coarse_tfidf = _tfidf_candidates(coarse_texts, n_label_candidates)
        coarse_cands = {c: coarse_kw.get(c) or coarse_tfidf.get(c, []) for c in coarse_texts}
    else:
        coarse_cands = _tfidf_candidates(coarse_texts, n_label_candidates)

    # Fine: representative title if requested, else keyword/TF-IDF
    if fine_rep_title_key:
        fine_rep = _representative_title_candidates(fine_nodes_map, G, vector_key, fine_rep_title_key)
        if keywords_key:
            fine_kw = _keyword_candidates(fine_nodes_map, G, keywords_key, n_label_candidates, keywords_sep)
            fine_tfidf = _tfidf_candidates(fine_texts, n_label_candidates)
            fine_fallback = {f: fine_kw.get(f) or fine_tfidf.get(f, []) for f in fine_texts}
        else:
            fine_fallback = _tfidf_candidates(fine_texts, n_label_candidates)
        fine_cands = {f: fine_rep.get(f) or fine_fallback.get(f, []) for f in fine_texts}
    elif keywords_key:
        fine_kw = _keyword_candidates(fine_nodes_map, G, keywords_key, n_label_candidates, keywords_sep)
        fine_tfidf = _tfidf_candidates(fine_texts, n_label_candidates)
        fine_cands = {f: fine_kw.get(f) or fine_tfidf.get(f, []) for f in fine_texts}
    else:
        fine_cands = _tfidf_candidates(fine_texts, n_label_candidates)

    coarse_samples = {c: [str(n) for n in ns] for c, ns in coarse_nodes_map.items()}
    fine_samples = {f: [str(n) for n in ns] for f, ns in fine_nodes_map.items()}

    if interactive:
        coarse_titles = _interactive_titles(
            coarse_samples, coarse_cands, level='coarse'
        )
        fine_titles = _interactive_titles(
            fine_samples, fine_cands, level='fine'
        )
    else:
        coarse_titles = _unique_titles(coarse_cands)
        fine_titles = _unique_titles(fine_cands)

    G.graph[f'{coarse_key}_titles'] = ';'.join(
        coarse_titles.get(i, '') for i in range(max(coarse_titles) + 1)
    )
    G.graph[f'{fine_key}_titles'] = ';'.join(
        fine_titles.get(i, '') for i in range(max(fine_titles) + 1)
    )

    return G


def compute_cluster_polygons(
    G: nx.Graph,
    rich_type: str = 'rich',
    cluster_key: str = 'c1',
    gap_radius: float | None = None,
    n_background: int = 50_000,
    seed: int = 42,
) -> dict[int, str]:
    """
    Compute a polygonal region for each cluster using a GMap-style Voronoi algorithm.

    Background points are scattered across a padded bounding box around the node layout.
    Those within gap_radius of any node are removed, creating a visual moat. A Voronoi
    diagram is then built from the remaining background points plus the node positions.
    The region for each cluster is the union of its members' Voronoi cells.

    Disjoint cluster regions (islands) are stored as separate polygon pieces joined by '///'.
    Within each piece, coordinates are space-separated and rounded to 3 decimal places.

    Parameters
    ----------
    G : nx.Graph
        Graph with 2-D positions ('x', 'y') and cluster IDs (cluster_key) on rich nodes.
    rich_type : str
        Value of the 'type' node attribute identifying rich nodes.
    cluster_key : str
        Node attribute key holding integer cluster IDs (as written by assign_clusters).
    gap_radius : float or None
        Minimum distance from any node that a background point must have to survive.
        Controls the width of the visual moat around each node. None autoscales to
        5 % of the bounding-box diagonal of the rich node positions.
    n_background : int
        Number of candidate background points sampled before gap filtering. More points
        give smoother polygon boundaries at the cost of compute time.
    seed : int
        Random seed for reproducible background point sampling.

    Returns
    -------
    dict[int, str]
        Maps each cluster id to its polygon string. Assign all levels to the graph with
        e.g. G.graph['c1_polygons'] = ';'.join(polygons[i] for i in sorted(polygons)).
    """
    rng = np.random.default_rng(seed)

    rich_nodes = [
        (n, d) for n, d in G.nodes(data=True)
        if d.get('type') == rich_type and cluster_key in d
    ]
    if not rich_nodes:
        return {}

    X = np.array([[d['x'], d['y']] for _, d in rich_nodes])
    memberships = np.array([int(d[cluster_key]) for _, d in rich_nodes])

    lo, hi = X.min(axis=0), X.max(axis=0)
    bbox_diag = float(np.linalg.norm(hi - lo))
    if gap_radius is None:
        gap_radius = 0.05 * bbox_diag

    # Sample background points in a padded bounding box
    margin = 0.2 * bbox_diag
    R = rng.uniform(lo - margin, hi + margin, (n_background, 2))

    # Remove points inside the moat around each node
    R = R[cdist(R, X).min(axis=1) >= gap_radius]

    vor = Voronoi(np.concatenate((X, R), axis=0))

    # Group node indices by cluster
    clusters: dict[int, list[int]] = {}
    for i, c in enumerate(memberships):
        clusters.setdefault(c, []).append(i)

    result: dict[int, str] = {}
    for c, indices in clusters.items():
        cell_polys = []
        for idx in indices:
            region = vor.regions[vor.point_region[idx]]
            if -1 in region or len(region) == 0:
                continue  # unbounded cell — shouldn't occur with enough background points
            cell_polys.append(Polygon(vor.vertices[region]))

        if not cell_polys:
            result[c] = ""
            continue

        union = unary_union(cell_polys)

        # Normalise to a list of simple polygons, handling disjoint islands
        pieces = list(union.geoms) if isinstance(union, MultiPolygon) else [union]

        result[c] = "///".join(
            " ".join(str(v) for v in np.round(np.array(poly.exterior.coords).reshape(-1), 3))
            for poly in pieces
        )

    return result


_CUSTOM_STOP_WORDS = frozenset({
    'seminar', 'research', 'participants', 'workshop', 'paper', 'work',
    'approach', 'method', 'based', 'using', 'new', 'also', 'used',
    'result', 'results', 'report', 'summary', 'program', 'dagstuhl',
    'present', 'study', 'studies', 'propose', 'proposed', 'address',
    'conference', 'commons', 'supply', 'semiconductor',
})

_STOP_WORDS = list(ENGLISH_STOP_WORDS | _CUSTOM_STOP_WORDS)

# Words that should be fully uppercased rather than title-cased
_ACRONYMS = frozenset({
    'ai', 'ml', 'nlp', 'rna', 'dna', 'hci', 'ui', 'ux', 'vr', 'ar',
    'iot', 'gpu', 'cpu', 'api', 'sat', 'smt', 'qbf', '3d', '2d',
})


def _fix_case(phrase: str) -> str:
    return ' '.join(
        w.upper() if w.lower() in _ACRONYMS else w.capitalize()
        for w in phrase.split()
    )


def _tfidf_titles(texts_per_cluster: dict[int, str], n_phrases: int = 1) -> dict[int, str]:
    """Return the top TF-IDF bigram (preferred) or unigram as a title for each cluster."""
    cluster_ids = sorted(texts_per_cluster)
    docs = [texts_per_cluster[c] for c in cluster_ids]

    if len(docs) < 2:
        words = [w for w in docs[0].split() if w.lower() not in _CUSTOM_STOP_WORDS]
        return {cluster_ids[0]: _fix_case(' '.join(words[:2]))} if cluster_ids else {}

    vec = TfidfVectorizer(
        stop_words=_STOP_WORDS,
        ngram_range=(1, 2),
        max_features=5000,
        token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b',  # exclude numbers and single chars
    )
    try:
        tfidf = vec.fit_transform(docs)
    except ValueError:
        return {c: '' for c in cluster_ids}

    feature_names = np.array(vec.get_feature_names_out())
    is_bigram = np.array([len(f.split()) == 2 for f in feature_names])

    titles = {}
    for i, c in enumerate(cluster_ids):
        scores = tfidf[i].toarray().ravel()
        # Prefer the highest-scoring bigram; fall back to unigram if none scored
        bigram_scores = np.where(is_bigram, scores, -1.0)
        if bigram_scores.max() > 0:
            best = feature_names[np.argmax(bigram_scores)]
        else:
            best = feature_names[np.argmax(scores)]
        phrases = [best]
        if n_phrases > 1:
            # Pick additional top phrases (bigram-preferred) that differ from chosen ones
            chosen = {best}
            remaining = np.argsort(scores)[::-1]
            for j in remaining:
                if len(phrases) >= n_phrases:
                    break
                if feature_names[j] not in chosen:
                    phrases.append(feature_names[j])
                    chosen.add(feature_names[j])
        titles[c] = ' / '.join(_fix_case(p) for p in phrases)
    return titles


def _unique_titles(cands_per_cluster: dict[int, list[str]]) -> dict[int, str]:
    """Pick the top candidate per cluster, skipping any already claimed by a prior cluster."""
    used: set[str] = set()
    titles: dict[int, str] = {}
    for c in sorted(cands_per_cluster):
        for cand in cands_per_cluster.get(c, []):
            if cand.lower() not in used:
                titles[c] = cand
                used.add(cand.lower())
                break
        else:
            titles[c] = cands_per_cluster[c][0] if cands_per_cluster.get(c) else ''
    return titles


def _genre_candidates(
    nodes_per_cluster: dict[int, list],
    G: nx.Graph,
    genre_key: str,
    n_candidates: int = 10,
) -> dict[int, list[str]]:
    """Return the top-n most frequent ', '-delimited genres per cluster."""
    from collections import Counter
    candidates: dict[int, list[str]] = {}
    for c, nodes in nodes_per_cluster.items():
        counter: Counter = Counter()
        for node in nodes:
            raw = G.nodes[node].get(genre_key, '')
            for genre in str(raw).split(','):
                genre = genre.strip()
                if genre:
                    counter[genre] += 1
        candidates[c] = [genre for genre, _ in counter.most_common(n_candidates)]
    return candidates


def _keyword_candidates(
    nodes_per_cluster: dict[int, list],
    G: nx.Graph,
    keywords_key: str,
    n_candidates: int = 10,
    sep: str = '; ',
) -> dict[int, list[str]]:
    """Return the top-n most frequent delimited keywords per cluster."""
    from collections import Counter
    candidates: dict[int, list[str]] = {}
    for c, nodes in nodes_per_cluster.items():
        counter: Counter = Counter()
        for node in nodes:
            raw = G.nodes[node].get(keywords_key, '')
            for kw in str(raw).split(sep):
                kw = kw.strip()
                if kw:
                    counter[kw] += 1
        candidates[c] = [kw for kw, _ in counter.most_common(n_candidates)]
    return candidates


def _representative_title_candidates(
    nodes_per_cluster: dict[int, list],
    G: nx.Graph,
    vector_key: str,
    title_key: str,
) -> dict[int, list[str]]:
    """Return a single-element list containing the title of the node closest to each cluster centroid."""
    candidates: dict[int, list[str]] = {}
    for c, nodes in nodes_per_cluster.items():
        vecs, titled_nodes = [], []
        for node in nodes:
            v = G.nodes[node].get(vector_key)
            t = G.nodes[node].get(title_key, '')
            if v is not None and t:
                vecs.append(np.asarray(v, dtype=np.float64))
                titled_nodes.append((node, t))
        if not vecs:
            candidates[c] = []
            continue
        mat = np.stack(vecs)
        centroid = mat.mean(axis=0)
        dists = np.linalg.norm(mat - centroid, axis=1)
        best_title = titled_nodes[int(np.argmin(dists))][1]
        candidates[c] = [best_title]
    return candidates


def _tfidf_candidates(
    texts_per_cluster: dict[int, str], n_candidates: int = 10
) -> dict[int, list[str]]:
    """Return the top n_candidates TF-IDF phrases (bigram-preferred) per cluster."""
    cluster_ids = sorted(texts_per_cluster)
    docs = [texts_per_cluster[c] for c in cluster_ids]

    if len(docs) < 2:
        words = [w for w in docs[0].split() if w.lower() not in _CUSTOM_STOP_WORDS]
        phrase = _fix_case(' '.join(words[:2])) if words else ''
        return {cluster_ids[0]: [phrase]} if cluster_ids else {}

    vec = TfidfVectorizer(
        stop_words=_STOP_WORDS,
        ngram_range=(1, 2),
        max_features=5000,
        token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b',
    )
    try:
        tfidf = vec.fit_transform(docs)
    except ValueError:
        return {c: [] for c in cluster_ids}

    feature_names = np.array(vec.get_feature_names_out())
    is_bigram = np.array([len(f.split()) == 2 for f in feature_names])

    candidates: dict[int, list[str]] = {}
    for i, c in enumerate(cluster_ids):
        scores = tfidf[i].toarray().ravel()
        # Rank: bigrams first by score, then unigrams
        bigram_order = np.argsort(np.where(is_bigram, scores, -1.0))[::-1]
        unigram_order = np.argsort(np.where(~is_bigram, scores, -1.0))[::-1]
        seen: set[str] = set()
        ranked: list[str] = []
        for idx in list(bigram_order) + list(unigram_order):
            if scores[idx] <= 0:
                continue
            phrase = _fix_case(feature_names[idx])
            if phrase not in seen:
                seen.add(phrase)
                ranked.append(phrase)
            if len(ranked) >= n_candidates:
                break
        candidates[c] = ranked
    return candidates


def _interactive_titles(
    samples_per_cluster: dict[int, list[str]],
    candidates_per_cluster: dict[int, list[str]],
    level: str,
) -> dict[int, str]:
    """Prompt the user to pick a label for each cluster from pre-computed candidates."""
    titles: dict[int, str] = {}

    for c in sorted(samples_per_cluster):
        candidates = candidates_per_cluster.get(c, [])
        samples = samples_per_cluster.get(c, [])
        sample_preview = ', '.join(f'"{s}"' for s in samples[:5])
        n_nodes = len(samples)

        print(f'\n--- {level.capitalize()} cluster {c}  ({n_nodes} nodes) ---')
        if sample_preview:
            print(f'  Samples: {sample_preview}')
        print('  Top label candidates:')
        for rank, phrase in enumerate(candidates, start=1):
            marker = ' *' if rank == 1 else ''
            print(f'    [{rank:2d}] {phrase}{marker}')
        default = candidates[0] if candidates else ''
        prompt = f'  Enter 1-{len(candidates)}, or type a custom label [default: {default!r}]: '
        while True:
            raw = input(prompt).strip()
            if raw == '':
                titles[c] = default
                break
            if raw.isdigit():
                idx = int(raw) - 1
                if 0 <= idx < len(candidates):
                    titles[c] = candidates[idx]
                    break
                print(f'  Please enter a number between 1 and {len(candidates)}.')
            else:
                titles[c] = raw
                break

    return titles


def _best_ward_k(X: np.ndarray, k_min: int = 2, k_max: int = 15) -> np.ndarray:
    """Ward clustering with silhouette-optimal k in [k_min, min(k_max, n-1)]."""
    best_labels = np.zeros(len(X), dtype=int)
    best_sil = -1.0
    for k in range(k_min, min(k_max, len(X) - 1) + 1):
        labels = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X)
        try:
            sil = silhouette_score(X, labels)
        except ValueError:
            continue
        if sil > best_sil:
            best_sil = sil
            best_labels = labels.copy()
    return best_labels
