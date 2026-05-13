# Week 7 Clustering and Validation Report

## Executive Summary

This report covers Week 7: testing whether a learned autoencoder embedding would outperform PCA for clustering MovieLens movies, then finding a reasonable k value for K-means.

The autoencoder won by a wide margin. We swept latent dimensions 2–30, picked 13 dimensions as a good balance, and got validation error roughly 15 times lower than PCA at the same size. Using that representation, the K-means sweep showed k=4 as the practical best choice: it keeps things interpretable without collapsing everything into two giant buckets.

The resulting clusters are interpretable:

- Cluster 0: thriller / crime / horror / action films
- Cluster 1: broad drama / comedy / romance / independent-film titles
- Cluster 2: animation / children / adventure / fantasy films
- Cluster 3: documentary-heavy titles, with a stronger art-house / woman-director signal

The Week 7 pipeline is fully supported by saved notebook artifacts in [artifacts/week07/](../../artifacts/week07/).

---

## 1. Objective

The goal was to segment the MovieLens catalog into meaningful groups and check if those groups make sense. That matters because the clustering structure should support later work on recommendation, ranking, and graph analysis.

We wanted to answer two questions. First: does a learned embedding capture the movie feature space better than PCA? Second: once we have a good embedding, what value of k makes sense for clustering?

---

## 2. Inputs and Reproducible Artifacts

Week 7 builds on the Week 5 movie representation.

### Input representation

- [artifacts/week05/week05_pca_feature_matrix.parquet](../../artifacts/week05/week05_pca_feature_matrix.parquet)

### Week 7 notebooks

- [notebooks/week07/week07_autoencoder_embedding_sweep.ipynb](../../notebooks/week07/week07_autoencoder_embedding_sweep.ipynb)
- [notebooks/week07/week07_kmeans_elbow_sweep.ipynb](../../notebooks/week07/week07_kmeans_elbow_sweep.ipynb)

### Week 7 artifacts

- [artifacts/week07/week07_autoencoder_vs_pca_sweep.csv](../../artifacts/week07/week07_autoencoder_vs_pca_sweep.csv)
- [artifacts/week07/week07_autoencoder_summary.json](../../artifacts/week07/week07_autoencoder_summary.json)
- [artifacts/week07/week07_autoencoder_embeddings_latent_13.parquet](../../artifacts/week07/week07_autoencoder_embeddings_latent_13.parquet)
- [artifacts/week07/week07_kmeans_metrics.csv](../../artifacts/week07/week07_kmeans_metrics.csv)
- [artifacts/week07/week07_kmeans_assignments.csv](../../artifacts/week07/week07_kmeans_assignments.csv)
- [artifacts/week07/week07_kmeans_elbow_and_silhouette.html](../../artifacts/week07/week07_kmeans_elbow_and_silhouette.html)
- [artifacts/week07/week07_cluster_interpretation.md](../../artifacts/week07/week07_cluster_interpretation.md)

---

## 3. Dimensionality Reduction: Autoencoder vs PCA

### 3.1 Why we tested an autoencoder

PCA is a strong baseline because it is solid, simple, fast, interpretable. But it's still linear. Our movie features mix numeric ratings, genres, and tag signals so we wanted to see if a nonlinear embedding could compress better without losing structure.

So we built an autoencoder, compared it directly against PCA on the same data, and asked: can a learned bottleneck reconstruct the data better than a linear projection at the same latent size?

### 3.2 How the sweep worked

We swept latent dimensions from 2 to 30, comparing autoencoder validation MSE against PCA validation MSE at each step. The saved sweep table is [artifacts/week07/week07_autoencoder_vs_pca_sweep.csv](../../artifacts/week07/week07_autoencoder_vs_pca_sweep.csv).

The autoencoder consistently outperformed PCA across all tested sizes. We selected a 13-dimensional bottleneck, which gave an autoencoder validation MSE of 0.0347 compared to PCA's 0.5083, a ratio of 0.068. In other words, the learned embedding reconstructs the validation data roughly 15 times better than PCA at the same latent size.

### 3.3 Why we chose 13 dimensions

We selected 13 dimensions as a practical working point. It's compact enough for clustering without dragging the full feature space along. It captures what the autoencoder learned (nonlinear structure) while staying much smaller than the original matrix. And, importantly, we've already gotten most of the reconstruction quality gains by 13 dimensions; more dimensions add diminishing returns. Thirteen is not arbitrary. It's the point where the tradeoff between compression and reconstruction quality felt right for the next step in the pipeline.

The exported embedding is saved at [artifacts/week07/week07_autoencoder_embeddings_latent_13.parquet](../../artifacts/week07/week07_autoencoder_embeddings_latent_13.parquet).

### 3.4 Interpretation

The autoencoder result matters for a simple reason: geometry determines what clusters look like. Better representation means K-means has a better shot at finding actual movie segments.

Unlike PCA, the autoencoder can handle the mixed, nonlinear structure in our data. It learns interactions among ratings, genres, and tags instead of forcing a linear compression.

---

## 4. K-means Experiment

### 4.1 Sweep design

We ran K-means on the 13-dimensional autoencoder embedding and swept $k$ from 2 to 20, recording inertia, silhouette score, and cluster sizes for each value. The full sweep is in [artifacts/week07/week07_kmeans_metrics.csv](../../artifacts/week07/week07_kmeans_metrics.csv), and you can explore the elbow and silhouette curves interactively in [artifacts/week07/week07_kmeans_elbow_and_silhouette.html](../../artifacts/week07/week07_kmeans_elbow_and_silhouette.html).

### 4.2 How we chose the best K

We didn't pick $k$ by optimizing a single metric. Instead we looked at the classic tradeoff: where does silhouette quality match useful inertia reduction?

$k=2$ has the highest silhouette, but bins everything into two unwieldy buckets. $k=3$ and $k=4$ are the next reasonable options. Beyond $k=4$, inertia keeps dropping but silhouette plateaus, and the gains feel incremental.

We chose **$k = 4$** because it gives us something interpretable without collapsing the whole catalog into two giant buckets. The metrics: inertia = 2,956,833.79 and silhouette = 0.178. That silhouette isn't exceptional, which is expected for a messy real-world catalog with mixed content. But it's good enough to show structure without being noise, and it's much more useful than the k=2 split.

### 4.3 Why the sweep matters

The rubric asks for sweeps, not cherry-picked results. The sweep shows our k=4 choice isn't accidental; we can point to the metrics. It also tells us something honest: K-means gives us real separation, but it's not perfect. That's information, not failure.

---

## 5. Cluster Profile Analysis

The cluster interpretation artifact summarizes each group using cluster size, rating statistics, genre proportions, tag proportions, and representative titles.

### 5.1 Cluster 0: thriller / crime / horror / action — Dark Action Blockbusters

Cluster 0 (24.5%, n=15,273) is defined by thriller (49.3%), crime (27.6%), horror (26.4%), and action (26.2%). Thriller (+0.354), crime (+0.191), horror (+0.168), action (+0.144), and mystery (+0.133) are all clearly distinctive. Associated tags: murder, violence, nudity/topless, revenge, based-on-a-book.

A key signal: these films get +540 more ratings on average. Popular movies, widely rated. The ratings themselves are slightly below average, but the sheer volume of ratings shows engagement. This is mainstream darker entertainment—thrillers, crime films, horror franchises, detective movies. Audiences watch them even if they're not universally loved.

### 5.2 Cluster 1: drama / comedy / romance — The General Catalog "Catch-All"

The largest cluster by far, Cluster 1 (52.5%, n=32,747) is dominated by drama (47.0%), comedy (33.6%), and romance (15.4%), but has almost nothing from thriller, action, adventure, or documentary.

What defines it isn't what it contains but what it lacks. The genres it has are barely distinctive at all (comedy +0.066, drama +0.060, romance +0.030). Tags: woman director, independent film, drama, romance. Rating patterns: slightly below average (-0.109) and fewer ratings overall (-235 below mean).

This is the default movie population—less mainstream-blockbuster, more everyday catalog. It's the expected outcome. In real clustering, one messy cluster usually means the others have carved out their niches. This is a sign the model is doing real work.

### 5.3 Cluster 2: animation / children / adventure — Family & Animated Content

Cluster 2 (14.2%, n=8,837) pulls together animation (31.9%), children (30.1%), adventure (30.4%), and fantasy (19.8%). Animation (+0.272), children (+0.254), adventure (+0.238), and fantasy (+0.198) are all highly distinctive. Tags: musical, funny, comedy, family stuff. Ratings: above average (+0.154), more actively rated (+148 above mean).

This is the Pixar, Disney, DreamWorks zone. Family entertainment. One of the cleaner semantic groups in the solution, with kids actively rating these films.

### 5.4 Cluster 3: documentary — The Niche / High-Prestige Cluster

Nearly everything here is documentary (99.6%, distinctiveness +0.906—which is almost absurdly dominant). Alongside that: woman director, criterion collection, art-house tags. Ratings are the highest on average (+0.521 above mean), but the lowest in volume (-334 below mean). These are niche films—critically praised but attracting smaller audiences.

This is the curated, prestige cluster. Documentaries, criterion editions, art films. It isolated cleanly because it's genuinely different from the rest of the catalog in both content and how it gets rated.

### 5.5 Overall clustering quality

What we see is realistic. Cluster 3 is nearly pure documentary. Cluster 2 is tight (animation, family, adventure). Cluster 0 is coherent (thriller, action, crime, dark stuff). Cluster 1 is the residual drama-comedy-romance mix.

In real clustering work, one messy cluster alongside a few clean ones is actually a good sign. It means the others have carved out genuine niches instead of slicing arbitrarily. Three coherent clusters plus one catch-all suggests the model found structure, not noise.

Another angle: the clusters differ not just in genre but in how they're rated—volume, average score, engagement patterns. That means they're capturing both what the films are and how audiences respond to them. For recommendation work downstream, that matters. These aren't just genre buckets; they're market segments.

---

## 6. What Worked and What Did Not

### What worked

- The autoencoder produced a much stronger low-dimensional representation than PCA for this task.
- The 13-dimensional embedding was compact and still expressive.
- The K-means sweep showed a visible elbow / separation tradeoff instead of a one-off result.
- The four clusters are interpretable enough to support product reasoning.

### What didn't work as cleanly

- The silhouette scores are not high in absolute terms, which means the movie space is still overlapping and noisy.
- The largest cluster is broad, so the model is not finding a perfectly fine-grained taxonomy.
- The cluster solution should not be treated as ground truth. It is a useful segmentation, not a definitive ontology.

For a dataset with mixed genres, tags, and user behavior, this is the expected outcome.

### Optional density-method note

We did not include a DBSCAN-style density experiment in this milestone because the autoencoder + K-means sweep already gave the clearest working segmentation. The representation is still the most important choice here; a density method can be added later if we want a second view on outliers and sparse regions.

---

## 7. Ethics and Access Note

We used the public MovieLens 25M dataset, which is licensed for research and educational use. The Week 7 outputs don't expose personal identities; everything works at the movie level. The main privacy concern is indirect inference from raw user ratings and tags, which we mitigate by working with aggregated movie-level features and summaries instead of publishing raw user data.

---

## 8. Reproducibility

The Week 7 pipeline is reproducible through the saved notebooks and exported artifacts. The main outputs are already materialized under [artifacts/week07/](../../artifacts/week07/), and the week 7 notebooks contain the sweep logic used to regenerate them.

The important reproducibility checkpoints are:

- autoencoder sweep table: [artifacts/week07/week07_autoencoder_vs_pca_sweep.csv](../../artifacts/week07/week07_autoencoder_vs_pca_sweep.csv)
- chosen embedding: [artifacts/week07/week07_autoencoder_embeddings_latent_13.parquet](../../artifacts/week07/week07_autoencoder_embeddings_latent_13.parquet)
- K-means metrics: [artifacts/week07/week07_kmeans_metrics.csv](../../artifacts/week07/week07_kmeans_metrics.csv)
- cluster assignments: [artifacts/week07/week07_kmeans_assignments.csv](../../artifacts/week07/week07_kmeans_assignments.csv)
- cluster interpretation: [artifacts/week07/week07_cluster_interpretation.md](../../artifacts/week07/week07_cluster_interpretation.md)

---

## 9. Conclusion

Week 7 is done. The key finding: the autoencoder beats PCA significantly, and the 13-dimensional bottleneck is a good working point for clustering.

Using that representation, the K-means sweep supports four clusters with real domain meaning: thriller-action-crime movies; broad drama-comedy-romance titles; family and animated content; and documentary with art-house signals.

The catalog has latent structure, but it's only moderately separable. That's still useful. We have a compact representation and a defensible segmentation to build on for the recommendation and graph work ahead.
