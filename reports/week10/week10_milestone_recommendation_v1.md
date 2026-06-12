# Week 10 Milestone Report: Recommendation, Ranking, and Predictive Decision Engine

## Executive Summary

This report covers Week 10: building a recommendation and ranking system for the MovieLens 25M catalog.
The work connects the technical infrastructure from Weeks 3, 5, and 7 (ingestion, feature representation,
and clustering) to a concrete decision task: *given a movie, which movies should be recommended next?*

Three systems were built and evaluated offline using a genre-relevance protocol over 4,994 query movies:

| System | Type | Signal | NDCG@10 |
|--------|------|--------|---------|
| `popularity_global` | Baseline | Bayesian rating count | **0.8390** |
| `content_cosine` | Baseline | Cosine similarity on AE-13 embeddings | **0.9925** |
| `svd_collaborative` | Advanced | Item factors from Truncated SVD (k=50) | **0.9496** |

The content-based cosine model achieved the highest NDCG@10 (0.9925), driven by the dense,
genre-coherent structure of the Week 7 autoencoder embedding space. The SVD collaborative model
ranks second, demonstrating that behavioral co-occurrence provides a strong signal complementary
to content features.

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

---

## 3. What Kind of System Is This?

This is a **ranking and discovery system** — not a user-to-item recommender.

- Given a seed movie, it ranks the catalog by relevance using one of three signals.
- The cluster layer from Week 7 provides secondary context (cluster-aware popularity) without extra modeling.
- Appropriate for cold-start catalog discovery: no per-user history is required.

The Week 10 framework answers:
> "You watched *The Matrix*. You might also like…"

---

## 4. Baseline Systems

### 4.1 Global popularity baseline

Every movie is ranked by its Bayesian-adjusted score:

$$\text{bayesian\_score} = \frac{C \cdot \mu + n \cdot \bar{r}}{C + n}$$

**Parameters:** global mean $\mu = 3.0714$, prior count $C = 1.0$ (10th percentile of rating counts).

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

- **Query movies**: 4,994–4,999 most-rated movies (≥1 rating, with valid genre labels)
- **Candidate pool**: top-20 recommendations from each system per query
- **Relevance criterion**: a recommendation is relevant if it shares **at least one genre** with the query

### 7.2 Relevance criterion note

Genre overlap is a reproducible, catalog-level proxy that requires no user ground truth.
It is conservative: two movies can share a genre without being close stylistically,
and can be stylistically close without sharing a genre label.
A user-history holdout evaluation is stronger; Section 8b adds a Leave-One-Out evaluation over user histories.

> **Recall@K limitation:** In the genre-based protocol, the `Recall@K` denominator was the number of relevant items within the system's own candidate pool — not the full catalog. This inflated Recall artificially (e.g. `content_cosine` Recall@20 = 0.9994). The defect is documented in the `recall_at_k()` function. The LOO evaluation in Section 8b uses Hit Rate@K as the correct recall proxy.

### 7.3 Metrics

| Metric | Definition |
|--------|-----------|
| **Precision@K** | Fraction of top-K recommendations that are genre-relevant |
| **Recall@K** | Fraction of genre-relevant items retrieved in top-K |
| **NDCG@K** | Normalized Discounted Cumulative Gain — position-weighted relevance |
| **Hit Rate@K** | 1 if ≥1 relevant item in top-K, else 0 |

---

## 8. Evaluation Results

### 8.1 Full metric table

| System | K | Precision@K | Recall@K | NDCG@K | Hit Rate@K |
|--------|---|-------------|----------|--------|------------|
| popularity_global | 5 | 0.5820 | 0.3621 | 0.8437 | 0.9553 |
| popularity_global | 10 | 0.5554 | 0.6089 | 0.8390 | 0.9720 |
| popularity_global | 20 | 0.4593 | 0.9808 | 0.8206 | 0.9808 |
| content_cosine | 5 | 0.9822 | 0.2535 | **0.9930** | **0.9982** |
| content_cosine | 10 | 0.9775 | 0.5032 | **0.9925** | **0.9990** |
| content_cosine | 20 | **0.9722** | **0.9994** | **0.9916** | **0.9994** |
| svd_collaborative | 5 | 0.8807 | 0.2656 | 0.9525 | 0.9890 |
| svd_collaborative | 10 | 0.8603 | 0.5155 | 0.9496 | 0.9958 |
| svd_collaborative | 20 | 0.8379 | 0.9986 | 0.9462 | 0.9986 |

> **Best values per column and K are bolded.**
> Source: `artifacts/week10/week10_evaluation_results.csv` (n_queries = 4,994).

**Key finding**: `content_cosine` dominates on all metrics. Its Precision@5 of 0.982 means
that 49 out of every 50 recommendations at rank 1–5 share at least one genre with the query.
This reflects the quality of the Week 7 autoencoder, which learned a compact 13-dim space that
is highly genre-coherent.

`svd_collaborative` beats `popularity_global` on Precision@K but trails `content_cosine`,
confirming that behavioral signals add value but content features are stronger under this
genre-based evaluation protocol.

### 8.2 Multi-metric comparison chart

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

> Source: `artifacts/week10/week10_loo_evaluation_results.csv` (seed=42, 10,000 users).

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

---

## 10. What Worked and What Did Not

### What worked

- The autoencoder embedding space from Week 7 transfers cleanly to item-to-item recommendation,
  achieving Precision@10 = 0.978 — near-perfect genre-relevant retrieval.
- The SVD factorization adds behavioral signal absent from content features:
  it correctly clusters behavioral co-occurrences even for genre-pure niche films.
- The cluster layer from Week 7 provides a useful secondary signal for cluster-aware popularity.
- The evaluation protocol is fully reproducible without user-level ground truth.

### What did not work

- **Popularity global**: systematically fails for Documentary, Western, Musical, and Film-Noir —
  any genre underrepresented in the global top-100 will receive zero relevant recommendations.
- **SVD coverage**: 45,871 movies (77.7% of the catalog) are excluded from the filtered matrix
  (< 50 ratings). Their item factors would be too noisy to produce meaningful recommendations.
- **Genre-based evaluation**: favors content methods by construction. SVD discovers behavioral
  patterns that may be more useful to users even when genres don't align perfectly.

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

All outputs are saved to `artifacts/week10/` (31 files, ~40 MB total).

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

Week 10 is complete. Three recommendation systems were built, evaluated, and compared under two complementary protocols:

**Genre-relevance evaluation** (item-to-item, 4,994 queries):
- **Content cosine** (NDCG@10 = **0.993**) is the strongest system for catalog discovery, leveraging the dense autoencoder embedding space from Week 7.
- **SVD collaborative** (NDCG@10 = 0.950) ranks second, capturing behavioral co-occurrence patterns absent from content features.
- **Popularity global** (NDCG@10 = 0.839) is the weakest baseline — genre-blind and fails for niche genres.

**Leave-One-Out evaluation** (user history, 10,000 users — Section 8b):
- **SVD collaborative** (Hit Rate@10 = **7.9%**) wins clearly, nearly doubling Popularity (4.6%) and more than doubling Content cosine (3.7%).
- This reversal reveals that SVD is the stronger *predictive* model for user behavior, while content similarity is better for catalog discovery.
- A Recall@K defect in the genre protocol (pool-denominator inflation) was identified and documented.

The recommendation layer is now a functioning component of the MovieLens Discovery pipeline.
The cluster segmentation from Week 7 underpins the cluster-aware baseline and sets up
the graph analytics layer in Week 12.
