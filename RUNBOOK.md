# Runbook: Rebuilding the MovieLens Discovery System

This is the single, ordered path from a fresh clone of this repository to every artifact referenced
in the milestone reports (`reports/week03` through `reports/week14`) and the `web/` demo. It exists
as the reproducibility deliverable for the Week 14 final integrated delivery
(`reports/week14/week14_milestone_final_integrated_delivery_v1.md`).

Each stage lists: what it needs from the previous stage, the exact command(s), and what it produces.
Stages 1–2 are one-command, `argparse`-driven scripts. Stages 3–5 are notebooks — run them
top-to-bottom in a fresh kernel; they are linear and do not depend on out-of-order cell execution,
but they are not wrapped in a CLI script (a known gap, tracked in Section 15 of the Week 14 report).

---

## 0. Prerequisites

```bash
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt
```

For the web demo (Stage 6), you also need [Bun](https://bun.sh/) (or Node.js 18+ with `npm`/`pnpm`
as a substitute for the `bun` commands below).

---

## 1. Week 3 — Ingestion and Cleaning

**Needs**: nothing (downloads raw data itself, or reuses `data/raw/ml-25m/` if already present).

```bash
# First time (downloads + extracts + cleans):
python3 scripts/build_week03_pipeline.py

# If data/raw/ml-25m/ already exists:
python3 scripts/build_week03_pipeline.py --skip-download
```

**Produces**: `data/processed/week03_v1/{movies_catalog,ratings_clean,tags_clean,movie_genres}.parquet`,
`data/processed/week03_v1/week03_cleaning_report.json`,
`data/interim/week03_{eda_summary,scale_metrics}.json`.

---

## 2. Week 5 — Feature Matrix and PCA

**Needs**: `data/processed/week03_v1/` from Stage 1.

```bash
python3 scripts/build_week05_pipeline.py
```

**Produces**: `artifacts/week05/week05_movie_feature_frame.parquet`,
`artifacts/week05/week05_pca_*` (scores, feature matrix, summary, thresholds, plots).

---

## 3. Week 7 — Autoencoder Embedding and K-means Clustering

**Needs**: `artifacts/week05/week05_pca_feature_matrix.parquet` from Stage 2.

Run both notebooks top-to-bottom, in this order:

1. `notebooks/week07/week07_autoencoder_embedding_sweep.ipynb`
2. `notebooks/week07/week07_kmeans_elbow_sweep.ipynb`

**Produces**: `artifacts/week07/week07_autoencoder_embeddings_latent_13.parquet`,
`artifacts/week07/week07_kmeans_assignments.csv`, plus the sweep/validation tables and plots
referenced in `reports/week07/`.

---

## 4. Week 10 — Recommenders and Offline Evaluation

**Needs**: `artifacts/week07/week07_autoencoder_embeddings_latent_13.parquet`,
`artifacts/week07/week07_kmeans_assignments.csv`, and
`data/processed/week03_v1/{ratings_clean,movies_catalog}.parquet`.

Run in this order:

1. `notebooks/week10/week10_baseline_recommender.ipynb` — popularity + content-cosine baselines
2. `notebooks/week10/week10_matrix_factorization.ipynb` — SVD collaborative filtering
3. `notebooks/week10/week10_offline_evaluation.ipynb` — genre-relevance + LOO evaluation, error analysis

**Produces**: `artifacts/week10/week10_{popularity_global,popularity_cluster}.csv`,
`week10_content_recs_top20.parquet`, `week10_svd_{item_factors,recs_top20}.parquet`,
`week10_{evaluation_results,loo_evaluation_results,error_analysis}.csv`.

---

## 5. Week 12 — Graph Construction and Centrality

**Needs**: `artifacts/week10/week10_svd_item_factors.parquet`,
`artifacts/week10/{week10_popularity_global.csv,week10_svd_recs_top20.parquet}`,
`artifacts/week07/week07_kmeans_assignments.csv`.

```text
notebooks/week12/week12_graph_analytics.ipynb
```

**Produces**: `artifacts/week12/week12_graph_metrics.parquet`,
`artifacts/week12/week12_graph_meta.json`, `artifacts/week12/week12_sensitivity_sweep.csv`,
`artifacts/week12/movie_graph_viz.json` (the sampled 400-node subgraph consumed by the web demo).

---

## 6. Web Demo

**Needs**: `artifacts/week10/` (for the recommender) and `artifacts/week12/movie_graph_viz.json`
(for the graph viewer).

### 6.1 Regenerate the demo data (only needed if the underlying artifacts changed)

```bash
env/bin/python scripts/build_recommender_catalog.py
cp artifacts/week12/movie_graph_viz.json web/public/movie_graph.json
```

### 6.2 Optional: fetch poster art for the graph viewer

```bash
export TMDB_API_KEY=your_v3_api_key   # https://www.themoviedb.org/settings/api
env/bin/python scripts/fetch_posters.py
```

The graph viewer works without this step; movies simply render without poster art.

### 6.3 Run the app

```bash
cd web
bun install
bun run dev
```

Open `http://localhost:4321`:

- `/` — 3D force-directed graph viewer (Week 12 subgraph, colored by Week 7 cluster).
- `/recommender` — search-and-recommend UI over the Week 10 `content_cosine` and `svd_collaborative`
  systems.

For a production build: `bun run build` (runs `astro check && astro build`), then `bun run preview`.

---

## What Would Change if the Raw Data Were Refreshed

- Stage 1's row counts, sparsity, and quality-check results (`week03_cleaning_report.json`) would
  change first.
- Every downstream stage would need a full rerun in order (2 → 3 → 4 → 5 → 6.1) — there is no
  incremental/partial rebuild path today. Feature scaling (Stage 2), the autoencoder weights and
  cluster assignments (Stage 3), the SVD factorization and both recommendation systems (Stage 4),
  and the graph's node set and edges (Stage 5, since nodes are gated on the ≥50-rating floor) are
  all derived from the ratings matrix and would all shift.
- The web demo's static JSON (Stage 6.1) is a projection of Stage 4/5 outputs and must be
  regenerated last.
