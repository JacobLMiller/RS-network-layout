"""
experiment_pl.py

Fixes p = 50 PCA dimensions and sweeps ℓ (Laplacian dims) from 0 to 100.

Hypothesis:
  - ℓ = 0  → pure dimensionality reduction (semantic-only input to UMAP)
  - ℓ > 0  → Laplacian structural signal is added, pulling co-linked events together
  - As ℓ grows, Spearman ρ(layout, semantic) decreases monotonically and
    ρ(layout, structural) increases monotonically.

Ground-truth distances
  - Semantic : cosine distance in the full sentence-embedding space (fixed, independent of ℓ)
  - Structural: 1 − cosine_similarity(B, B) where B is the biadjacency matrix (co-attendance)

Outputs (in experiment_pl/)
  - results.csv           — ρ values per configuration
  - correlation_plot.pdf  — line chart of the two ρ curves
  - layouts_grid.pdf      — small-multiples of rich-node layouts coloured by semantic axis
"""

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from umap import UMAP

from embed.preprocess import preprocess

# ── Configuration ─────────────────────────────────────────────────────────────
GRAPHML     = 'dagstuhl-filtered.graphml'
RICH_TYPE   = 'event'
SPARSE_TYPE = 'person'
DATA_KEY    = 'seminar_summary'
VECTOR_KEY  = 'vector'

P_DIM    = 50                                          # fixed PCA dimensions
L_MAX    = 100                                         # maximum Laplacian dims
L_VALUES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # ℓ sweep; p fixed at P_DIM

N_PAIRS   = 30_000   # subsampled pairs for Spearman (speeds up 1431²/2 ≈ 1M pairs)
UMAP_SEED = 42

OUT_DIR = 'experiment_pl'
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load & preprocess ──────────────────────────────────────────────────────
print("Loading graph …")
G = nx.read_graphml(GRAPHML)
G.graph['name'] = 'dagstuhl-filtered'
rich_count   = sum(1 for _, d in G.nodes(data=True) if d.get('type') == RICH_TYPE)
sparse_count = sum(1 for _, d in G.nodes(data=True) if d.get('type') == SPARSE_TYPE)
print(f"  {rich_count:,} events, {sparse_count:,} persons, {G.number_of_edges():,} edges")

print("\nPreprocessing (sentence embeddings, cached) …")
preprocess(G, rich_type=RICH_TYPE, data_key=DATA_KEY, data_format='text',
           vector_key=VECTOR_KEY, verbose=True)

# ── 2. Extract ordered arrays ─────────────────────────────────────────────────
rich_nodes   = [n for n, d in G.nodes(data=True) if d.get('type') == RICH_TYPE]
sparse_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == SPARSE_TYPE]
N = len(rich_nodes)

raw_vecs = np.array([G.nodes[n][VECTOR_KEY] for n in rich_nodes])  # (N, D)

# ── 3. Ground-truth semantic distances ───────────────────────────────────────
print("\nComputing semantic ground-truth distances (cosine in full embedding space) …")
D_sem = cosine_distances(raw_vecs)   # (N, N)

# ── 4. Ground-truth structural distances (B B^T) ─────────────────────────────
print("Computing structural ground-truth distances (B cosine similarity) …")
B = nx.bipartite.biadjacency_matrix(
    G, row_order=rich_nodes, column_order=sparse_nodes
).toarray().astype(float)

row_norms = np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-10)
B_norm = B / row_norms
D_struct = 1.0 - (B_norm @ B_norm.T)   # cosine distance in co-attendance space

# ── 5. Subsample pairs once (same pairs across all ℓ for comparability) ───────
rng     = np.random.default_rng(0)
flat_i  = rng.integers(0, N, size=N_PAIRS * 2)
flat_j  = rng.integers(0, N, size=N_PAIRS * 2)
mask    = flat_i != flat_j
flat_i, flat_j = flat_i[mask][:N_PAIRS], flat_j[mask][:N_PAIRS]

d_sem_ref    = D_sem[flat_i, flat_j]
d_struct_ref = D_struct[flat_i, flat_j]

# ── 6. Pre-compute full PCA and Laplacian eigenvectors once ───────────────────
print("\nPCA (fitting to full semantic space) …")
pca_full = PCA(n_components=P_DIM).fit(raw_vecs)
X_pca    = pca_full.transform(raw_vecs)          # (N, P_DIM)

print("Computing normalised Laplacian eigenvectors …")
S = B @ B.T
deg = np.maximum(S.sum(axis=1), 1e-10)
d_inv_sqrt = 1.0 / np.sqrt(deg)
L_norm = np.eye(N) - (d_inv_sqrt[:, None] * S * d_inv_sqrt[None, :])
_, eigvecs_all = eigh(L_norm, subset_by_index=[1, L_MAX])  # (N, L_MAX)

# Colour axis for layout plots: first PCA component (fixed reference)
color_vals = X_pca[:, 0]

# ── 7. Sweep ─────────────────────────────────────────────────────────────────
results = []
all_layouts = {}

for ell in L_VALUES:
    print(f"\np={P_DIM}, ℓ={ell:3d} …", flush=True)

    lap_part = eigvecs_all[:, :ell] if ell > 0 else np.empty((N, 0))
    X        = np.hstack([X_pca, lap_part])   # (N, P_DIM + ℓ)

    Y = UMAP(n_neighbors=15, min_dist=1e-2,
             random_state=UMAP_SEED).fit_transform(X)
    all_layouts[ell] = Y

    d_layout = np.sqrt(((Y[flat_i] - Y[flat_j]) ** 2).sum(axis=1))

    rho_sem,    _ = spearmanr(d_layout, d_sem_ref)
    rho_struct, _ = spearmanr(d_layout, d_struct_ref)
    print(f"  ρ_sem={rho_sem:.4f}   ρ_struct={rho_struct:.4f}", flush=True)

    results.append({'p': P_DIM, 'ell': ell,
                    'rho_sem': rho_sem, 'rho_struct': rho_struct})

# ── 8. Save layouts and node index ───────────────────────────────────────────
# layouts.npz: keys "ell_{ell}" → (N, 2) float32 array; row order matches node_index.csv
npz_payload = {f'ell_{ell}': all_layouts[ell].astype(np.float32) for ell in L_VALUES}
npz_path = f'{OUT_DIR}/layouts.npz'
np.savez(npz_path, **npz_payload)
print(f"\nSaved {npz_path}  (keys: {list(npz_payload)})")

# node_index.csv: row_idx → node_id + selected metadata for easy rejoining
NODE_ATTRS = ['seminar_number', 'seminar_name', 'seminar_keywords', 'seminar_classification']
index_rows = []
for row_idx, node_id in enumerate(rich_nodes):
    attrs = G.nodes[node_id]
    row = {'row_idx': row_idx, 'node_id': node_id}
    for key in NODE_ATTRS:
        row[key] = attrs.get(key, '')
    index_rows.append(row)

index_df  = pd.DataFrame(index_rows)
index_path = f'{OUT_DIR}/node_index.csv'
index_df.to_csv(index_path, index=False)
print(f"Saved {index_path}  ({len(index_df)} rows)")

# ── 9. Save results CSV ───────────────────────────────────────────────────────
df = pd.DataFrame(results)
csv_path = f'{OUT_DIR}/results.csv'
df.to_csv(csv_path, index=False)
print(f"\nResults:\n{df.to_string(index=False)}")
print(f"\nSaved {csv_path}")

# ── 10. Correlation line plot ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(df['ell'], df['rho_sem'],    'o-',  color='steelblue',  lw=2, label='ρ(layout, semantic)')
ax.plot(df['ell'], df['rho_struct'], 's--', color='tomato',     lw=2, label='ρ(layout, structural)')
ax.set_xlabel('ℓ  (Laplacian dims;  p = 50 fixed)')
ax.set_ylabel('Spearman  ρ')
ax.set_title('Layout fidelity as ℓ increases  (p = 50 fixed)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(df['ell'])
fig.tight_layout()
corr_path = f'{OUT_DIR}/correlation_plot.pdf'
fig.savefig(corr_path, dpi=150)
print(f"Saved {corr_path}")
plt.close(fig)

# ── 11. Layout grid ───────────────────────────────────────────────────────────
ncols = 6
nrows = (len(L_VALUES) + ncols - 1) // ncols   # ceiling division → 2 rows
fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.6 * nrows))
axes_flat = axes.flat

vmin, vmax = color_vals.min(), color_vals.max()

for ax, ell in zip(axes_flat, L_VALUES):
    Y  = all_layouts[ell]
    sc = ax.scatter(Y[:, 0], Y[:, 1], c=color_vals, cmap='RdYlBu',
                    vmin=vmin, vmax=vmax, s=3, alpha=0.7, linewidths=0)
    ax.set_title(f'p=50, ℓ={ell}', fontsize=10)
    ax.axis('off')

for ax in list(axes_flat)[len(L_VALUES):]:   # hide unused panels
    ax.set_visible(False)

fig.colorbar(sc, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.015,
             pad=0.02, label='1st semantic PCA component')
fig.suptitle('Dagstuhl layout sweep  (p = 50 fixed, ℓ = 0 … 100)\ncoloured by semantic axis', y=1.01)
fig.tight_layout()
grid_path = f'{OUT_DIR}/layouts_grid.pdf'
fig.savefig(grid_path, bbox_inches='tight', dpi=150)
print(f"Saved {grid_path}")
plt.close(fig)

print("\nDone.")
