import networkx as nx

from embed.preprocess import preprocess
from embed.semantic_layout import Semantic
from embed.cluster import assign_clusters, compute_cluster_polygons
from embed._utils import save_graph

GRAPHML = 'netflix-small.graphml'
OUTPUT  = 'outputs_new/netflix-small/Semantic.json'

RICH_TYPE   = 'movie'
SPARSE_TYPE = 'user'
DATA_KEY    = 'data'

print("Loading graph …")
G = nx.read_graphml(GRAPHML)
G.graph['name'] = 'netflix-small'
rich_count   = sum(1 for _, d in G.nodes(data=True) if d.get('type') == RICH_TYPE)
sparse_count = sum(1 for _, d in G.nodes(data=True) if d.get('type') == SPARSE_TYPE)
print(f"  {rich_count:,} movie nodes, {sparse_count:,} user nodes, {G.number_of_edges():,} edges")

print("\nPreprocessing (sentence embeddings) …")
preprocess(G, rich_type=RICH_TYPE, data_key=DATA_KEY, data_format='text',
           vector_key='vector', verbose=True)

print("\nRunning Semantic layout (PCA + Laplacian + UMAP) …")
pos = Semantic(G, rich_type=RICH_TYPE, sparse_type=SPARSE_TYPE,
               vector_key='vector', verbose=1).layout(pca_dim=90, laplacian_dims=10, n_neighbors=50)

for node, (x, y) in pos.items():
    G.nodes[node]['x'] = float(x)
    G.nodes[node]['y'] = float(y)
print(f"  Positioned {len(pos):,} nodes.")

print("\nAssigning clusters …")
assign_clusters(
    G,
    rich_type=RICH_TYPE,
    data_key=DATA_KEY,
    k_coarse=8,
    k_max_fine=10,
    keywords_key='keywords',
    keywords_sep=', ',
    coarse_genre_key='genres',
    fine_rep_title_key='title',
    vector_key='vector',
    interactive=False,
)
print("  Done.")

print("\nComputing cluster polygons …")
poly_c1 = compute_cluster_polygons(G, rich_type=RICH_TYPE, cluster_key='c1')
poly_c2 = compute_cluster_polygons(G, rich_type=RICH_TYPE, cluster_key='c2')
print(f"  {len(poly_c1)} coarse regions, {len(poly_c2)} fine regions.")

print(f"\nSaving to {OUTPUT} …")
save_graph(G, OUTPUT, polygons={'c1': poly_c1, 'c2': poly_c2})
print("Done.")
