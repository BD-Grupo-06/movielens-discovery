# Week 10 Milestone Report: Recommendation, Ranking, and Predictive Decision Engine

## Executive Summary

This report covers Week 10: building a recommendation and ranking system for the MovieLens 25M catalog.
The work connects the technical infrastructure from Weeks 3, 5, and 7 (ingestion, feature representation,
and clustering) to a concrete decision task: *given a movie, which movies should be recommended next?*

Four systems were built and evaluated offline under two complementary protocols: a corrected
genre-relevance protocol over 4,994 query movies, and a Leave-One-Out (LOO) protocol over
10,000 user histories:

| System | Type | Signal | NDCG@10 (genre, corrected) | Hit Rate@10 (LOO) |
|--------|------|--------|---------------------------|-------------------|
| `popularity_global` | Baseline | Raw rating-count popularity | 0.5852 | 0.0465 |
| `cluster_popularity` | Hybrid baseline | Week 7 segmentation → Bayesian ranking | 0.5590 | — (see §8b) |
| `content_cosine` | Baseline | Cosine similarity on AE-13 embeddings | **0.9797** | 0.0368 |
| `svd_collaborative` | Advanced | Item factors from Truncated SVD (k=50) | 0.8715 | **0.0795** |

The content-based cosine model dominates the genre-relevance protocol (NDCG@10 = 0.9797),
driven by the dense, genre-coherent structure of the Week 7 autoencoder embedding space.
Under the LOO protocol the ranking reverses: the SVD collaborative model is the strongest
*predictive* system (Hit Rate@10 = 7.95%). Two metric defects in the original genre protocol
(pool-denominator Recall and self-referential ideal DCG) were identified, documented, and
**corrected by recomputation** (`scripts/build_week10_eval_corrected.py`); Section 8 reports
both the legacy and the corrected numbers for traceability.

---

## 1. Objective

The goal was to connect the prior technical layers to a product-relevant decision task:

> *Given a seed movie, rank the catalog by relevance to that movie.*

This is a **ranking and discovery system** built on top of the Week 7 clustering layer.
It is item-to-item (not user-to-item), which is appropriate for catalog discovery without a logged-in
user context.

Specific questions answered:
- Does collaborative filtering (SVD) beat content-based similarity on genre-relevant recommendations?
- Where do the failure cases of each system lie?
- How does the cluster layer from Week 7 support the recommendation layer?

---

## 2. Inputs and Reproducible Artifacts

### Input artifacts from previous weeks

| Artifact | Source | Purpose |
|----------|--------|---------|
| `week07_autoencoder_embeddings_latent_13.parquet` | Week 7 | Content-based embedding (13-dim) |
| `week07_kmeans_assignments.csv` | Week 7 | Cluster-aware popularity (k=7) |
| `ratings_clean.parquet` | Week 3 | SVD and popularity signal (25M ratings) |
| `movies_catalog.parquet` | Week 3 | Movie metadata and genre labels (59,047 movies) |

### Week 10 notebooks

| Notebook | Purpose |
|----------|---------|
| [week10_baseline_recommender.ipynb](../../notebooks/week10/week10_baseline_recommender.ipynb) | Popularity + content-based baselines |
| [week10_matrix_factorization.ipynb](../../notebooks/week10/week10_matrix_factorization.ipynb) | SVD collaborative filtering |
| [week10_offline_evaluation.ipynb](../../notebooks/week10/week10_offline_evaluation.ipynb) | Evaluation protocol, metrics, error analysis |

### Week 10 output artifacts

| Artifact | Description | Size |
|----------|-------------|------|
| `week10_popularity_global.csv` | Global Bayesian popularity ranking (59,047 movies) | 4.8 MB |
| `week10_popularity_cluster.csv` | Cluster-aware popularity ranking | 5.1 MB |
| `week10_content_recs_top20.parquet` | Content-based top-20 per query movie | 0.6 MB |
| `week10_svd_item_factors.parquet` | SVD item latent factors (13,176 × 50) | 2.8 MB |
| `week10_svd_recs_top20.parquet` | SVD collaborative top-20 per query movie | 0.5 MB |
| `week10_evaluation_results.csv` | Full metric comparison table | 0.5 KB |
| `week10_error_analysis.csv` | Strong and failure cases per system | 2.4 KB |
| `week10_evaluation_summary.json` | JSON summary of all evaluation results (includes LOO) | 2.5 KB |
| `week10_loo_evaluation_results.csv` | Leave-One-Out metric comparison table | 0.5 KB |
| `week10_loo_evaluation_comparison.png` | LOO bar chart (Hit Rate, Precision, NDCG @ K) | ~90 KB |
| `week10_genre_eval_corrected.csv` | Corrected genre-protocol metrics, 4 systems (incl. hybrid) | 1.2 KB |
| `week10_data_alignment.csv` | Layer-by-layer data alignment for the hybrid claim | 0.5 KB |

### Week 10 scripts

| Script | Purpose |
|--------|---------|
| `scripts/build_week10_eval_corrected.py` | Recomputes the genre evaluation with corrected Recall/NDCG, adds the `cluster_popularity` hybrid, and emits the data-alignment table — from committed artifacts only (no raw data needed) |

---

## 3. What Kind of System Is This?

Against the four candidate framings — recommendation, ranking, prediction, or segmentation
feeding ranking — this project is, precisely:

> **An item-to-item recommendation system, operationalized as a ranking task, with one
> variant that is segmentation feeding ranking, and a prediction-style evaluation.**

| Framing | Where it appears in this project |
|---------|----------------------------------|
| **Recommendation** | The product task: given a seed movie, suggest movies ("You watched *The Matrix*. You might also like…"). Item-to-item, not user-to-item — appropriate for cold-start catalog discovery with no logged-in user. |
| **Ranking** | The operationalization: every system outputs an *ordered* top-20 list, and all evaluation metrics (Precision@K, NDCG@K, Hit Rate@K) are ranking metrics. |
| **Segmentation feeding ranking** | The `cluster_popularity` hybrid: the Week 7 k-means segmentation (k=7) defines the candidate pool, and the Bayesian popularity score ranks within it. Section 4.3 documents the data alignment this requires. |
| **Prediction** | The LOO evaluation (Section 8b) reframes the system as a predictive task: can it predict the *next* movie a user actually rated? |

The primary identity is **recommendation via ranking**; segmentation (Week 7) feeds the
hybrid baseline; prediction is the lens of the second evaluation protocol, not the system's
design goal.

---

## 4. Baseline Systems

### 4.1 Global popularity baseline

Every movie is ranked by its Bayesian-adjusted score:

$$\text{bayesian\_score} = \frac{C \cdot \mu + n \cdot \bar{r}}{C + n}$$

**Parameters:** global mean $\mu = 3.0714$, prior count $C = 1.0$ (10th percentile of rating counts).

> **Note on the evaluated list:** the published artifact `week10_popularity_global.csv` ranks by
> `bayesian_score`, but the static top-20 list used in the genre evaluation ranks by **raw rating
> count** (most-rated movies). Both are reported here for transparency; the reproduction check in
> `build_week10_eval_corrected.py` confirms the rating-count list is the one behind the published
> metrics (max abs. difference 0.0002 vs the original run).

The catalog has a strong long-tail: 59,047 movies, but the top-5% account for the overwhelming
majority of rating volume, as shown in the distribution below.

![Rating count distribution (log10 scale)](../../artifacts/week10/week10_popularity_distribution.png)

### 4.2 Cluster-aware popularity baseline

Restricts the ranking to movies within the same Week 7 cluster as the query movie (k=7).
The cluster statistics show a pronounced engagement disparity: Cluster 4 (23,093 movies)
has a mean rating count of only 9.1 (low-engagement catalog tail), while Cluster 0
(mainstream Drama/Comedy) averages 1,666 ratings per movie.

| Cluster | Movies | Mean rating count | Mean avg. rating | Mean Bayesian score |
|---------|--------|------------------|-----------------|-------------------|
| 0 | 9,363 | 1,665.9 | 3.291 | 3.288 |
| 1 | 2,650 | 505.1 | 3.130 | 3.152 |
| 2 | 5,013 | 327.6 | 3.064 | 3.091 |
| 3 | 10,036 | 463.6 | 2.739 | 2.815 |
| 4 | 23,093 | 9.1 | 3.042 | 3.061 |
| 5 | 5,346 | 55.4 | 3.383 | 3.321 |
| 6 | 3,546 | 356.1 | 3.121 | 3.126 |

![Cluster-level popularity statistics](../../artifacts/week10/week10_cluster_popularity_stats.png)

This hybrid is evaluated alongside the other three systems in Section 8.1b.

### 4.3 Hybrid claim and data alignment

`cluster_popularity` is a **hybrid system**: an unsupervised *content* segmentation (Week 7
k-means over autoencoder embeddings of Week 5 content/behavioral features) feeding a
*behavioral* popularity ranking (Week 3 ratings). A hybrid is only valid if the layers actually
join — the table below documents the alignment, all on the `movieId` key
(source: `artifacts/week10/week10_data_alignment.csv`):

| Layer | Join key | Movies | Catalog coverage |
|-------|----------|-------:|-----------------:|
| `movies_catalog` (Week 3) | movieId | 62,423 | 100.00% |
| Autoencoder embeddings AE-13 (Week 7) | movieId | 62,423 | 100.00% |
| K-means assignments, k=7 (Week 7) | movieId | 62,423 | 100.00% |
| Rated movies with genres = evaluation universe (Week 10) | movieId | 59,047 | 94.59% |
| Cluster popularity ranking (Week 10 hybrid) | movieId | 59,047 | 94.59% |
| SVD item factors (Week 10) | movieId | 13,176 | 21.11% |
| Query set, content/SVD (Week 10) | movieId | 5,000 | 8.01% |
| Evaluated queries (intersection, with genres) | movieId | 4,994 | 8.00% |

Where rows drop, and why:

- **62,423 → 59,047**: movies with zero ratings have no popularity signal and leave the
  evaluation universe. Every rated movie has a cluster assignment, so the hybrid join loses
  nothing beyond unrated movies.
- **59,047 → 13,176**: the SVD filter (≥ 50 ratings/movie) — a coverage limitation of the
  collaborative system, not of the hybrid.
- **5,000 → 4,994**: the evaluation intersects the content and SVD query sets and requires at
  least one genre label, leaving exactly **4,994** queries (this resolves the 4,994 vs 4,999
  count discrepancy between earlier artifact versions: 4,999 was counted before the
  intersection).

---

## 5. Content-Based Baseline — Cosine Similarity on Autoencoder Embeddings

### 5.1 Method

Retrieves the top-20 most similar movies using cosine similarity over the
13-dimensional autoencoder embedding space from Week 7 (embedding_dim=13,
59,047 movies with ratings have embeddings).

$$\text{sim}(i, j) = \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{\|\mathbf{e}_i\| \cdot \|\mathbf{e}_j\|}$$

### 5.2 Similarity distribution

The top-1 cosine similarity across 5,000 query movies is concentrated near 1.0,
indicating a dense, well-structured embedding space where genre-coherent neighborhoods
form clearly.

![Top-1 cosine similarity distribution](../../artifacts/week10/week10_cosine_similarity_dist.png)

---

## 6. Advanced System — SVD Collaborative Filtering

### 6.1 Model design

Factorizes the user-item rating matrix using Truncated SVD:

$$\mathbf{R}_{\text{centered}} \approx \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top$$

**Matrix**: 162,540 users × 13,176 movies (after filtering: ≥ 50 ratings/movie, ≥ 20 ratings/user).
**Density**: 1.15% non-zero entries. **Centering**: per-user mean subtraction.
**Latent dimension**: k=50, chosen from the sweep below.

### 6.2 Latent factor sweep

| Components | Explained variance | Time (s) |
|-----------|-------------------|----------|
| 10 | 9.33% | 1.35 |
| 20 | 12.27% | 2.11 |
| 30 | 14.42% | 2.07 |
| **50** | **17.79%** | **2.83** |
| 75 | 21.11% | 3.57 |
| 100 | 23.89% | 4.60 |
| 150 | 28.58% | 5.37 |

k=50 was selected: the variance curve flattens past this point with diminishing returns.

![SVD latent factor sweep](../../artifacts/week10/week10_svd_sweep.png)

### 6.3 Singular value spectrum

Fast decay in the singular value spectrum (first component σ₁=996, second σ₂=519)
confirms strong low-rank structure in the rating matrix — a favorable signal for
collaborative filtering.

![SVD singular value analysis](../../artifacts/week10/week10_svd_singular_values.png)

### 6.4 SVD score distribution

Top-1 SVD collaborative similarity scores, shown below. Scores are concentrated
at higher values than the content-based baseline, reflecting tight item-factor
alignment for popular movie clusters.

![SVD top-1 similarity score distribution](../../artifacts/week10/week10_svd_score_dist.png)

---

## 7. Offline Evaluation Protocol

### 7.1 Setup

- **Evaluation universe**: the 59,047 rated movies with genre labels (movies with zero
  ratings carry no signal for any system and are excluded)
- **Query movies**: exactly **4,994** — the intersection of the content and SVD query sets
  (5,000 most-rated movies each), restricted to queries with at least one genre label
- **Recommendation lists**: top-20 per system per query (the query itself is always excluded)
- **Relevance criterion**: a recommendation is relevant if it shares **at least one genre** with the query

### 7.2 Relevance criterion note

Genre overlap is a reproducible, catalog-level proxy that requires no user ground truth.
It is conservative: two movies can share a genre without being close stylistically,
and can be stylistically close without sharing a genre label.
A user-history holdout evaluation is stronger; Section 8b adds a Leave-One-Out evaluation over user histories.

> **Recall@K and NDCG@K corrections:** In the original run, the `Recall@K` denominator was the
> number of relevant items within the system's own top-20 pool — not the full universe — which
> inflated Recall artificially (e.g. `content_cosine` Recall@20 = 0.9994). The ideal DCG in
> `NDCG@K` had the same self-referential defect (it was built from the relevant count inside the
> retrieved top-K). Both defects are documented in the evaluation notebook **and corrected by
> recomputation** in `scripts/build_week10_eval_corrected.py`, which first reproduces the
> original numbers (max abs. difference 0.0002) and then recomputes `catalog_recall_at_k`
> (denominator = genre-relevant movies in the full universe; on average **22,301** per query)
> and `ndcg_at_k_corrected` (ideal DCG = K relevant items). Section 8.1 keeps the legacy table
> for traceability; Section 8.1b reports the corrected metrics. The LOO evaluation in Section 8b
> additionally provides a Recall that is correct by construction.

### 7.3 Metrics

| Metric | Definition |
|--------|-----------|
| **Precision@K** | Fraction of top-K recommendations that are genre-relevant |
| **Recall@K** | Fraction of genre-relevant items retrieved in top-K. *Legacy version (8.1)*: denominator = relevant items in the system's own pool (inflated). *Corrected version (8.1b)*: denominator = relevant items in the full universe |
| **NDCG@K** | Normalized Discounted Cumulative Gain — position-weighted relevance. *Corrected version*: ideal DCG assumes K relevant items |
| **Hit Rate@K** | 1 if ≥1 relevant item in top-K, else 0 |

With ~22,301 relevant movies per query and only 20 recommendations, corrected catalog Recall
is structurally tiny (≤ 20/22,301 ≈ 0.0009 on average) for *any* system; **Precision@K and
NDCG@K are the informative metrics under this protocol**, and Recall is meaningfully measured
only in the LOO protocol.

### 7.4 Candidate-pool definition

The rubric requires an explicit candidate-pool definition — what each system is allowed to
rank from, per protocol:

| System | Genre protocol (item-to-item) | LOO protocol (user histories) |
|--------|-------------------------------|-------------------------------|
| `popularity_global` | Static global list: top-20 movies by raw rating count, minus the query | Top-500 most-rated movies in the LOO training split, minus the user's seen movies |
| `cluster_popularity` | Movies in the query's Week 7 cluster (k=7), ranked by Bayesian score, minus the query | Not evaluated (see §8b) |
| `content_cosine` | All 59,047 rated movies with AE-13 embeddings, minus the query | The 13,176 SVD-factored movies, minus the user's seen movies |
| `svd_collaborative` | The 13,176 movies in the filtered SVD matrix (≥ 50 ratings), minus the query | The 13,176 SVD-factored movies, minus the user's seen movies |

Two asymmetries are deliberate and disclosed: (a) in the genre protocol, `svd_collaborative`
can only recommend from 22.3% of the catalog (its training filter), while `content_cosine`
ranks the full rated universe; (b) in the LOO protocol, only the 9,885 of 10,000 sampled users
whose held-out movie is inside the 13,176-movie factor space are scoreable for the content and
SVD systems (n_queries = 10,000 for popularity, 9,885 for the other two).

---

## 8. Evaluation Results

### 8.1 Legacy metric table (original run — Recall and NDCG inflated, kept for traceability)

| System | K | Precision@K | Recall@K (pool, inflated) | NDCG@K (legacy) | Hit Rate@K |
|--------|---|-------------|---------------------------|-----------------|------------|
| popularity_global | 5 | 0.5820 | 0.3621 | 0.8437 | 0.9553 |
| popularity_global | 10 | 0.5554 | 0.6089 | 0.8390 | 0.9720 |
| popularity_global | 20 | 0.4593 | 0.9808 | 0.8206 | 0.9808 |
| content_cosine | 5 | 0.9822 | 0.2535 | 0.9930 | 0.9982 |
| content_cosine | 10 | 0.9775 | 0.5032 | 0.9925 | 0.9990 |
| content_cosine | 20 | 0.9722 | 0.9994 | 0.9916 | 0.9994 |
| svd_collaborative | 5 | 0.8807 | 0.2656 | 0.9525 | 0.9890 |
| svd_collaborative | 10 | 0.8603 | 0.5155 | 0.9496 | 0.9958 |
| svd_collaborative | 20 | 0.8379 | 0.9986 | 0.9462 | 0.9986 |

> Source: `artifacts/week10/week10_evaluation_results.csv` (n_queries = 4,994).
> The Recall and NDCG columns carry the self-referential defects described in Section 7.2;
> read the corrected table below instead.

### 8.1b Corrected metric table (all four systems, including the hybrid)

| System | K | Precision@K | Catalog Recall@K | NDCG@K (corrected) | Hit Rate@K |
|--------|---|-------------|------------------|--------------------|------------|
| popularity_global | 5 | 0.5820 | 0.000136 | 0.6207 | 0.9553 |
| popularity_global | 10 | 0.5554 | 0.000263 | 0.5852 | 0.9720 |
| popularity_global | 20 | 0.4595 | 0.000455 | 0.5062 | 0.9808 |
| cluster_popularity | 5 | 0.5479 | 0.000151 | 0.5731 | 0.8104 |
| cluster_popularity | 10 | 0.5449 | 0.000297 | 0.5590 | 0.8214 |
| cluster_popularity | 20 | 0.5784 | 0.000634 | 0.5762 | 0.9864 |
| content_cosine | 5 | **0.9822** | **0.000296** | **0.9834** | **0.9982** |
| content_cosine | 10 | **0.9775** | **0.000588** | **0.9797** | **0.9990** |
| content_cosine | 20 | **0.9722** | **0.001168** | **0.9753** | **0.9994** |
| svd_collaborative | 5 | 0.8807 | 0.000251 | 0.8882 | 0.9890 |
| svd_collaborative | 10 | 0.8603 | 0.000485 | 0.8715 | 0.9958 |
| svd_collaborative | 20 | 0.8379 | 0.000934 | 0.8519 | 0.9986 |

> **Best values per column and K are bolded** — `content_cosine` leads every column.
> Source: `artifacts/week10/week10_genre_eval_corrected.csv` (n_queries = 4,994).
> Catalog Recall is structurally tiny under this protocol (Section 7.3) and is shown for
> completeness, not comparison against the legacy column.

**Key finding**: `content_cosine` still dominates after correction. Its Precision@5 of 0.982
means 49 out of every 50 recommendations at rank 1–5 share at least one genre with the query —
a direct consequence of the genre-coherent Week 7 autoencoder space. The correction's largest
effect is on `popularity_global`: its legacy NDCG@10 of 0.839 drops to 0.585 once the ideal
DCG stops adapting to the system's own output.

**Hybrid finding**: `cluster_popularity` trades coverage for depth. Restricting candidates to
the query's Week 7 cluster *lowers* Hit Rate@10 (0.821 vs 0.972 global — niche queries can
land in clusters whose popular movies don't share their genres), but its precision barely
decays with depth (0.548 → 0.578 from K=5 to K=20) while the global list collapses
(0.582 → 0.460). At K=20 the hybrid is the **more precise popularity baseline**, confirming
the Week 7 segmentation contributes real signal — segmentation feeding ranking, as claimed in
Section 3.

### 8.2 Multi-metric comparison chart

> The charts in 8.2–8.4 were rendered from the original (legacy) run and therefore show the
> pre-correction Recall/NDCG values for the three original systems; the corrected numbers are
> in the 8.1b table.

![Evaluation comparison across all systems and K values](../../artifacts/week10/week10_evaluation_comparison.png)

### 8.3 NDCG@10 summary

![NDCG@10 comparison across systems](../../artifacts/week10/week10_ndcg10_comparison.png)

### 8.4 Per-query NDCG@10 distribution

Box plots of per-query NDCG@10 reveal that `content_cosine` has both the highest median
and the tightest spread, while `popularity_global` shows the widest variance: its performance
depends entirely on whether the globally popular movies happen to match the query's genre.

![Per-query NDCG@10 distribution by system](../../artifacts/week10/week10_ndcg10_distribution.png)

---

## 8b. Leave-One-Out (LOO) Evaluation

To complement the genre-based protocol, a second evaluation was run using user rating histories from `ratings_clean.parquet`.

**Protocol:**
- Users with ≥ 5 ratings; 10,000 sampled (seed=42).
- Last item by timestamp hidden as test; remainder used as training history.
- Each system recommends top-K items excluding already-seen movies.
- Metric: **Hit Rate@K** (= Recall@K when there is exactly 1 test item per user) and **NDCG@K**.

> **Note on Precision@K in LOO:** With a single test item per user, `Precision@K = Hit Rate@K / K` by definition — it carries no additional information and should not be interpreted as low performance.

### LOO results

| System | Hit Rate@5 | Hit Rate@10 | Hit Rate@20 | NDCG@10 |
|--------|-----------|------------|------------|--------|
| `popularity_global` | 0.0287 | 0.0465 | 0.0767 | 0.0238 |
| `content_cosine` | 0.0219 | 0.0368 | 0.0563 | 0.0187 |
| **`svd_collaborative`** | **0.0515** | **0.0795** | **0.1221** | **0.0432** |

> Source: `artifacts/week10/week10_loo_evaluation_results.csv` (seed=42; n_queries = 10,000
> for popularity, 9,885 for content/SVD — see the candidate-pool definition in Section 7.4).
> `cluster_popularity` is not in this table: the LOO protocol requires the raw user histories
> (`ratings_clean.parquet`), which are reproducible from the Week 3 pipeline but not committed
> to the repository; re-running the LOO notebook with a per-user cluster-restricted pool is
> documented as follow-up work in Section 10.

**Key finding:** SVD collaborative **reverses the ranking** seen in the genre-based evaluation — it achieves nearly double the Hit Rate@10 of Popularity and more than double that of Content cosine. This confirms that SVD captures genuine behavioral predictive signal, while content-based similarity is stronger for catalog discovery but weaker at predicting individual user consumption.

The contrast also validates the Recall correction: the genre-based Recall@20 for `content_cosine` was 0.9994 (inflated by the pool-denominator bug), while the real retrieval rate under LOO is 5.6%.

![LOO evaluation comparison](../../artifacts/week10/week10_loo_evaluation_comparison.png)

---

## 9. Error Analysis

### 9.1 Strong cases (Precision@10 = 1.0)

| System | Query movie | Query genres | Behavior |
|--------|-------------|-------------|----------|
| `popularity_global` | Money Train (1995) | Action\|Comedy\|Crime\|Drama\|Thriller | Multi-genre query overlaps with mainstream top-100 |
| `popularity_global` | Dead Presidents (1995) | Action\|Crime\|Drama | Same — broad genre coverage in global list |
| `content_cosine` | Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy | Tight animation cluster in AE space |
| `content_cosine` | Grumpier Old Men (1995) | Comedy\|Romance | Dense comedy-romance neighborhood |
| `svd_collaborative` | GoldenEye (1995) | Action\|Adventure\|Thriller | Strong user co-occurrence with Action/Thriller |
| `svd_collaborative` | Tom and Huck (1995) | Adventure\|Children | Clear behavioral cluster among family films |

### 9.2 Failure cases (Hit Rate@10 = 0)

| System | Query movie | Query genres | Root cause |
|--------|-------------|-------------|-----------|
| `popularity_global` | Heidi Fleiss: Hollywood Madam (1995) | Documentary | Top-100 global list contains no documentaries |
| `popularity_global` | Anne Frank Remembered (1995) | Documentary | Same — documentary genre absent from global top |
| `content_cosine` | Fair Game (1995) | Action | Embedding conflates this action film with non-action neighbors |
| `content_cosine` | Repo Man (1984) | Comedy\|Sci-Fi | Mixed genre embedding pulls to incorrect neighborhood |
| `svd_collaborative` | Showgirls (1995) | Drama | Anomalous rating pattern; SVD factors are genre-incoherent |
| `svd_collaborative` | Wyatt Earp (1994) | Western | Very few Western films in filtered matrix → noisy item factors |

**Pattern**: The popularity baseline fails systematically for all niche genres (Documentary, Western,
Film-Noir, Musical) — these genres are simply not in the global top-100. Content and SVD fail on
edge cases where the catalog signal is weak (few ratings, genre-ambiguous films).

The corrected evaluation (8.1b) adds a hybrid-specific failure mode: `cluster_popularity`
fixes the popularity baseline's niche-genre failure *when the niche has its own cluster*
(Documentary queries now draw from Cluster 5, which is 99.9% documentaries), but inherits a
new one — queries whose genre is a minority within their cluster get recommendations matching
the cluster's majority genres instead, which is why its Hit Rate@10 (0.821) is below the
global baseline's (0.972).

---

## 10. What Worked and What Did Not

### What worked

- The autoencoder embedding space from Week 7 transfers cleanly to item-to-item recommendation,
  achieving Precision@10 = 0.978 — near-perfect genre-relevant retrieval.
- The SVD factorization adds behavioral signal absent from content features:
  it correctly clusters behavioral co-occurrences even for genre-pure niche films.
- The cluster layer from Week 7 provides measurable signal: the `cluster_popularity` hybrid is
  the more precise popularity baseline at K=20 (0.578 vs 0.460) and solves the niche-genre
  failure for genres with their own cluster (Section 9.2).
- The evaluation protocol is fully reproducible without user-level ground truth — and the
  corrected metrics can be recomputed from committed artifacts alone
  (`scripts/build_week10_eval_corrected.py`), with a built-in reproduction check against the
  original run.

### What did not work

- **Popularity global**: systematically fails for Documentary, Western, Musical, and Film-Noir —
  any genre underrepresented in the global top-100 will receive zero relevant recommendations.
- **SVD coverage**: 45,871 movies (77.7% of the catalog) are excluded from the filtered matrix
  (< 50 ratings). Their item factors would be too noisy to produce meaningful recommendations.
- **Genre-based evaluation**: favors content methods by construction. SVD discovers behavioral
  patterns that may be more useful to users even when genres don't align perfectly.
- **Original Recall/NDCG implementation**: both metrics were self-referential (denominators
  derived from the system's own output). Corrected by recomputation; legacy values retained
  in 8.1 for traceability.
- **Hybrid under LOO (follow-up)**: evaluating `cluster_popularity` under the LOO protocol
  requires regenerating `ratings_clean.parquet` (Week 3 pipeline) and re-running the LOO
  notebook with a cluster-restricted candidate pool; left as documented follow-up work.

---

## 11. Ethics and Access Note

- **Data source**: GroupLens MovieLens 25M public release (research license).
- **Access**: Educational and research use only; raw data not redistributed.
- **Personal data risk**: all user identifiers are anonymized integer IDs in the source dataset.
- **Mitigation**: all analysis operates at the movie level.
  SVD item factors are aggregate representations; no individual user behavior is exposed.
  No re-identification of users is attempted or possible.

---

## 12. Reproducibility

Run notebooks manually in this order:

1. `notebooks/week10/week10_baseline_recommender.ipynb`
2. `notebooks/week10/week10_matrix_factorization.ipynb`
3. `notebooks/week10/week10_offline_evaluation.ipynb`

Then run the corrected evaluation (works from committed artifacts alone — no raw data needed):

```bash
python3 scripts/build_week10_eval_corrected.py
```

The script validates itself by reproducing the original legacy metrics (it aborts if the
maximum absolute difference exceeds 0.002) before writing
`week10_genre_eval_corrected.csv`, `week10_data_alignment.csv`, and the
`genre_eval_corrected` block of `week10_evaluation_summary.json`.

All outputs are saved to `artifacts/week10/` (~40 MB total).

**Key parameters for reproducibility:**

| Parameter | Value |
|-----------|-------|
| Catalog size | 59,047 movies |
| Global mean rating μ | 3.0714 |
| Bayesian prior C | 1.0 |
| Embedding dimension | 13 |
| Clusters (k) | 7 |
| SVD min movie ratings | 50 |
| SVD min user ratings | 20 |
| SVD n_components | 50 |
| SVD explained variance | 17.79% |
| Matrix dimensions | 162,540 users × 13,176 movies |
| Matrix density | 1.15% |
| Random state | 42 |

---

## 13. Conclusion

Week 10 is complete. Four recommendation systems — two popularity baselines (one of them a
segmentation-fed hybrid), a content-based baseline, and an advanced collaborative model —
were built, evaluated, and compared under two complementary protocols:

**Genre-relevance evaluation, corrected** (item-to-item, 4,994 queries — Section 8.1b):
- **Content cosine** (corrected NDCG@10 = **0.980**) is the strongest system for catalog discovery, leveraging the dense autoencoder embedding space from Week 7.
- **SVD collaborative** (corrected NDCG@10 = 0.872) ranks second, capturing behavioral co-occurrence patterns absent from content features.
- **Popularity global** (corrected NDCG@10 = 0.585) is the weakest at depth; the **cluster popularity hybrid** trades hit coverage for depth-stable precision and is the better popularity baseline at K=20 — evidence that the Week 7 segmentation feeds the ranking with real signal.

**Leave-One-Out evaluation** (user history, 10,000 users — Section 8b):
- **SVD collaborative** (Hit Rate@10 = **7.9%**) wins clearly, nearly doubling Popularity (4.6%) and more than doubling Content cosine (3.7%).
- This reversal reveals that SVD is the stronger *predictive* model for user behavior, while content similarity is better for catalog discovery.
- The Recall@K and NDCG@K defects in the original genre protocol were identified, documented, and **corrected by recomputation** (Sections 7.2 and 8.1b), with the legacy numbers retained for traceability.

In rubric terms (Section 3): the project is an **item-to-item recommendation system
operationalized as ranking**, with **segmentation feeding ranking** in the hybrid baseline and
a **prediction**-framed second protocol. The recommendation layer is now a functioning
component of the MovieLens Discovery pipeline, and the cluster segmentation from Week 7 sets
up the graph analytics layer in Week 12.
