# RS-network-layout

Code accompanying the paper **"Visualization of Rich-Sparse Bipartite Networks"**.

Produces 2-D layouts of bipartite graphs (rich nodes × sparse nodes) using five methods and evaluates them with neighbourhood- and aesthetic-quality metrics.

---

## Repository structure

```
embed/                   Layout methods and shared utilities
  semantic_layout.py     Semantic layout (PCA → graph Laplacian → UMAP)
  common_dist.py         Union-graph methods (FR, UMAP)
  independent.py         Independent / Procrustes-based methods
  preprocess.py          Sentence-transformer embedding pipeline
  cluster.py             Two-level agglomerative clustering + polygon hulls
  clean.py               Graph pre-filtering utility
  similarity_matrices.py Semantic and Jaccard distance helpers
  _utils.py              I/O helpers (read_json_representation, save_graph, …)

metrics/                 Evaluation metrics (SNS, SNKL, NHit, NP, SF, ND, AR, ELV)

load_datasets/           Data-conversion scripts (raw → json_data/ or GraphML)
raw_data/                Raw dataset files (see Data section below)
json_data/               Preprocessed graphs in internal JSON format (gitignored)

embed_all.py             Run all layout methods on every graph in json_data/
evaluate.py              Evaluate all outputs/ layouts and write metric_out/
cluster_all.py           Assign clusters and polygons to existing outputs

run_dagstuhl_semantic.py Full pipeline: Dagstuhl GraphML → outputs_new/ (input for the online tool)
run_netflix_semantic.py  Full pipeline: Netflix GraphML  → outputs_new/

viz.ipynb                Generates all figures and tables in the paper
draw.ipynb               Generates all figures and tables in the paper

fetch_dagstuhl.py        Download Dagstuhl seminar data from the public API
fetch_netflix.py         Build netflix.graphml from the Netflix Prize zip files
fetch_descriptions.py    Enrich movie nodes with TMDB metadata
filter_netflix.py        Reduce netflix.graphml to a (p,q)-core subgraph

outputs/                 Layout JSON files (all methods, all datasets)
outputs_new/             Layout JSON files with cluster labels and polygon hulls
metric_out/              Evaluation results (JSON)
```

---

## Methods

| Key | Description |
|-----|-------------|
| `Semantic` | PCA on sentence embeddings → graph-Laplacian refinement → UMAP |
| `Union_FR` | Union-graph Fruchterman–Reingold |
| `Union_UMAP` | Union-graph UMAP |
| `Independent_R` | Independent bipartite embedding with Procrustes alignment |
| `Independent_S` | Independent bipartite embedding without alignment |
| `Naive_FR` | Baseline: standard spring layout (NetworkX) |

---

## Datasets

| Dataset | Nodes | Description |
|---------|-------|-------------|
| Dagstuhl | Seminars × Researchers | Schloss Dagstuhl seminar co-attendance graph |
| VisPub | Papers × Authors | IEEE VIS publications 1990–2024 |
| VAST MC3 | Companies × Contacts | VAST Challenge 2023 MC3 knowledge graph (companies and their contacts) |
| Netflix | Movies × Users | Netflix Prize subset enriched with TMDB metadata — download from [Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) |
| Instacart | Products × Customers | Instacart grocery orders — download from [Kaggle](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis) |

The Netflix and Instacart raw files are too large to include in this repository.
See the **Prepare data** section below for instructions on obtaining and preprocessing them.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Python 3.11+ recommended. GPU not required (sentence-transformer inference runs on CPU).

### 2. Prepare data

**VisPub / Dagstuhl / VAST MC3 (internal JSON format, used by `embed_all.py` / `evaluate.py`)**

Place the preprocessed JSON files in `json_data/`. These can be produced from the
GraphML sources in `raw_data/` using the scripts in `load_datasets/`:

```bash
python load_datasets/convert_to_json.py
```

**Dagstuhl GraphML (used by `run_dagstuhl_semantic.py`)**

```bash
python fetch_dagstuhl.py          # downloads dagstuhl.graphml
# dagstuhl-filtered.graphml is a pre-filtered version committed to the repo
```

**Netflix GraphML (used by `run_netflix_semantic.py`)**

Download the Netflix Prize data from [Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)
and place the zip files in `raw_data/netflix/` (see `raw_data/netflix/README`). Then:

```bash
python fetch_descriptions.py      # produces netflix_enriched.csv  (requires TMDB API key)
python fetch_netflix.py           # produces netflix.graphml
python filter_netflix.py          # produces netflix-filtered.graphml / netflix-small.graphml
```

**Instacart (internal JSON format)**

Download the Instacart Market Basket Analysis dataset from
[Kaggle](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis)
and place the CSV files in `raw_data/instacart/`. Then:

```bash
python load_datasets/convert_instacart_to_json.py
```

### 3. Run layouts

**All methods on all json_data/ graphs (writes to `outputs/`):**

```bash
python embed_all.py
```

**Semantic layout with clustering on a single GraphML (writes to `outputs_new/`):**

```bash
python run_dagstuhl_semantic.py
python run_netflix_semantic.py
```

### 4. Evaluate

```bash
python evaluate.py    # reads outputs/, writes metric_out/
```

---

## Outputs

`outputs/<dataset>/<Method>.json` — node-link JSON consumable by the front-end
visualisation (`index.html`).

`outputs_new/<dataset>/Semantic.json` — richer JSON including per-node cluster
labels (`c1`, `c2`) and polygon hull strings (`c1_polygons`, `c2_polygons`).

`metric_out/all_data.json` — aggregated evaluation metrics for all datasets and
methods.

---

## License

See [LICENSE](LICENSE).
