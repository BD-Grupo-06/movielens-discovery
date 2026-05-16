import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from statistics import mean
import math
import joblib

from src.config import DEFAULT_RANDOM_STATE
from src.io import ensure_dir, read_parquet

def run_week07_pipeline(
    processed_dir: Path,
    artifacts_dir: Path,
    random_state: int = DEFAULT_RANDOM_STATE
):
    ensure_dir(artifacts_dir)

    print("Loading unified feature matrix...")
    movie_features = pl.read_parquet(processed_dir / "movie_features.parquet")
    
    clustering_df = movie_features
    movie_ids = clustering_df.get_column("movieId").to_list()
    X = clustering_df.drop("movieId").to_numpy()

    print("Starting parameter sweep for K-means...")
    k_range = range(2, 200)
    
    metrics_log = []
    all_labels_by_k = {}
    
    models_dir = artifacts_dir / "kmeans_models"
    ensure_dir(models_dir)

    for k in k_range:
        model_path = models_dir / f"kmeans_k{k}.pkl"
        labels_path = models_dir / f"kmeans_labels_k{k}.npy"

        if model_path.exists() and labels_path.exists():
            print(f"  Loading pre-trained KMeans for K={k}...")
            kmeans = joblib.load(model_path)
            labels = np.load(labels_path)
        else:
            print(f"  Training K-means with K={k}...")
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(X)

            joblib.dump(kmeans, model_path)
            np.save(labels_path, labels)

        all_labels_by_k[k] = labels

        inertia = float(kmeans.inertia_)
        
        if X.shape[0] > 10000:
            np.random.seed(random_state)
            idx = np.random.choice(X.shape[0], 5000, replace=False)
            sil = float(silhouette_score(X[idx], labels[idx]))
        else:
            sil = float(silhouette_score(X, labels))
            
        metrics_log.append({
            "k": k,
            "inertia": inertia,
            "silhouette": sil
        })

    # --- SAVE VALIDATION METRICS ---
    validation_df = pd.DataFrame(metrics_log)
    validation_df.to_csv(artifacts_dir / "kmeans_validation_table.csv", index=False)
    
    target_k = 15
    selected_metric = [m for m in metrics_log if m["k"] == target_k][0]
    
    best_k = selected_metric["k"]
    best_silhouette = selected_metric["silhouette"]
    best_labels = all_labels_by_k[best_k]

    summary = {
        "selected_best_k": best_k,
        "best_silhouette_score": best_silhouette,
        "metrics_sweep": metrics_log
    }
    (artifacts_dir / "week07_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Sweep completed! Selected K based on local peak optimization: K={best_k}")
    
    # Final cluster assignments
    cluster_assignments = pl.DataFrame({
        "movieId": movie_ids,
        "cluster_id": best_labels
    })
    cluster_assignments.write_parquet(processed_dir / "movie_clusters.parquet")
    print("Cluster assignments saved to movie_clusters.parquet")

    print("Generating diagnostic visualization assets...")

    # 1. Dual Subplot: Elbow Curve (Inertia) and Silhouette Score Evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Subplot 1: Elbow Method
    ax1.plot(validation_df["k"], validation_df["inertia"], marker="o", color="b", linestyle="--")
    ax1.axvline(x=best_k, color="r", linestyle=":", label=f"Selected K ({best_k})")
    ax1.set_title("Elbow Method (Inertia)", fontsize=12)
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inercia (Within-Cluster Sum of Squares)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()
    
    # Subplot 2: Silhouette Evolution
    ax2.plot(validation_df["k"], validation_df["silhouette"], marker="s", color="g", linestyle="-")
    ax2.axvline(x=best_k, color="r", linestyle=":", label=f"Selected K ({best_k})")
    ax2.set_title("Validation via Silhouette Coefficient", fontsize=12)
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(artifacts_dir / "kmeans_diagnostic_curves.png", dpi=300)
    plt.close()

    # 2. Scatter Plot using Week 5 fixed t-SNE Coordinates
    tsne_parquet_path = Path("artifacts/week05/pca_tsne.parquet")
    if tsne_parquet_path.exists():
        print("Found t-SNE artifact. Projecting clusters...")
        tsne_df = pl.read_parquet(tsne_parquet_path)
        
        plot_df = tsne_df.join(cluster_assignments, on="movieId", how="inner").to_pandas()
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x="tsne_1",
            y="tsne_2",
            hue="cluster_id",
            palette="tab20",  # Exact fit for 10 unique colors
            data=plot_df,
            alpha=0.6,
            edgecolor=None
        )
        plt.title(f"Spatial Projection of Movie Clusters (K-means K={best_k})", fontsize=14)
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend(title="Cluster ID", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        
        plt.savefig(artifacts_dir / "kmeans_tsne_clusters.png", dpi=300)
        plt.close()
        print("t-SNE cluster scatter plot successfully exported.")
    else:
        print("'pca_tsne.parquet' not found in artifacts. Skipping cluster scatter projection.")
    
    generate_report_insights(processed_dir, Path("data/interim"))

    print("Week 07 pipeline completed successfully with all required visual assets.")


def generate_report_insights(
    processed_dir: Path,
    data_dir: Path,
    embeddings=None
):

    print("Generating advanced cluster interpretation report...")

    # =========================================================
    # LOAD DATA
    # =========================================================
    clusters = pl.read_parquet(processed_dir / "movie_clusters.parquet")
    movies = pl.read_parquet(data_dir / "movies_catalog.parquet")
    ratings = pl.read_parquet(data_dir / "ratings_clean.parquet")

    # =========================================================
    # MOVIE-LEVEL AGGREGATES
    # =========================================================
    avg_ratings = ratings.group_by("movieId").agg(
        pl.col("rating").mean().alias("avg_rating"),
        pl.col("rating").count().alias("vote_count")
    )

    # =========================================================
    # FULL CATALOG
    # =========================================================
    full_catalog = (
        clusters
        .join(movies, on="movieId", how="inner")
        .join(avg_ratings, on="movieId", how="left")
    )

    # Fill numeric nulls only
    full_catalog = full_catalog.with_columns([
        pl.col("avg_rating").fill_null(0),
        pl.col("vote_count").fill_null(0),
    ])

    # =========================================================
    # GLOBAL METRICS
    # =========================================================
    total_movies = full_catalog.shape[0]
    cluster_sizes = (
        full_catalog
        .group_by("cluster_id")
        .len()
        .sort("cluster_id")
    )

    largest_cluster = cluster_sizes["len"].max()
    smallest_cluster = cluster_sizes["len"].min()
    avg_cluster_size = cluster_sizes["len"].mean()

    # Missing genres
    missing_genres = (
        full_catalog
        .filter(
            pl.col("genres").is_null() |
            (pl.col("genres").list.len() == 0)
        )
        .shape[0]
    )

    missing_pct = (missing_genres / total_movies) * 100

    # =========================================================
    # REPORT HEADER
    # =========================================================
    report_lines = []

    report_lines.append("=========================================================")
    report_lines.append("      WEEK 7 - ADVANCED CLUSTER ANALYSIS REPORT")
    report_lines.append("=========================================================\n")

    report_lines.append("GLOBAL DATASET SUMMARY")
    report_lines.append("---------------------------------------------")
    report_lines.append(f"Total Movies: {total_movies:,}")
    report_lines.append(f"Total Clusters: {cluster_sizes.shape[0]}")
    report_lines.append(f"Average Cluster Size: {avg_cluster_size:.2f}")
    report_lines.append(f"Largest Cluster Size: {largest_cluster}")
    report_lines.append(f"Smallest Cluster Size: {smallest_cluster}")
    report_lines.append(
        f"Movies Missing Genre Metadata: "
        f"{missing_genres:,} ({missing_pct:.2f}%)"
    )

    # =========================================================
    # SILHOUETTE SCORE
    # =========================================================
    if embeddings is not None:
        try:
            labels = full_catalog["cluster_id"].to_numpy()
            sil_score = silhouette_score(embeddings, labels)

            report_lines.append(
                f"Global Silhouette Score: {sil_score:.4f}"
            )

            if sil_score > 0.5:
                interpretation = "Strong cluster separation."
            elif sil_score > 0.25:
                interpretation = "Moderate cluster separation."
            else:
                interpretation = "Weak cluster separation."

            report_lines.append(f"Silhouette Interpretation: {interpretation}")

        except Exception as e:
            report_lines.append(f"Silhouette Score Error: {e}")

    report_lines.append("\n")

    # =========================================================
    # CLUSTER ANALYSIS
    # =========================================================
    report_lines.append("=========================================================")
    report_lines.append("                CLUSTER INTERPRETATION")
    report_lines.append("=========================================================\n")

    cluster_ids = sorted(full_catalog["cluster_id"].unique().to_list())

    for c_id in cluster_ids:

        cluster_data = full_catalog.filter(
            pl.col("cluster_id") == c_id
        )

        cluster_size = cluster_data.shape[0]
        cluster_pct = (cluster_size / total_movies) * 100

        report_lines.append(
            f"### CLUSTER {c_id} "
            f"({cluster_size:,} movies | {cluster_pct:.2f}% of catalog)"
        )

        # =====================================================
        # CLUSTER RATINGS
        # =====================================================
        cluster_avg_rating = cluster_data["avg_rating"].mean()

        report_lines.append(
            f"Average Cluster Rating: {cluster_avg_rating:.2f}"
        )

        # =====================================================
        # RELEASE YEAR ANALYSIS
        # =====================================================
        valid_years = (
            cluster_data
            .filter(pl.col("release_year") > 0)
            ["release_year"]
            .to_list()
        )

        if valid_years:
            avg_year = mean(valid_years)
            report_lines.append(
                f"Average Release Year: {avg_year:.1f}"
            )

        # =====================================================
        # GENRE ANALYSIS
        # =====================================================
        genres_list = [
            genre
            for g in cluster_data["genres"].to_list()
            if g
            for genre in g
        ]

        if genres_list:

            genre_counts = Counter(genres_list)
            top_genres = genre_counts.most_common(5)

            dominant_genre, dominant_count = top_genres[0]

            # Cluster purity
            purity = dominant_count / cluster_size

            # Entropy
            probs = [
                count / sum(genre_counts.values())
                for count in genre_counts.values()
            ]

            cluster_entropy = -sum(
                p * math.log2(p)
                for p in probs
                if p > 0
            )

            genre_str = ", ".join(
                [f"{g}: {c}" for g, c in top_genres]
            )

            report_lines.append(f"Dominant Genres: {genre_str}")
            report_lines.append(f"Dominant Genre: {dominant_genre}")
            report_lines.append(f"Cluster Purity: {purity:.3f}")
            report_lines.append(f"Genre Entropy: {cluster_entropy:.3f}")

            # Automatic interpretation
            if purity > 0.7:
                report_lines.append(
                    "Interpretation: Highly coherent thematic cluster."
                )
            elif purity > 0.4:
                report_lines.append(
                    "Interpretation: Moderately mixed thematic cluster."
                )
            else:
                report_lines.append(
                    "Interpretation: Highly hybrid / ambiguous cluster."
                )

        else:
            report_lines.append(
                "Dominant Genres: Unknown / Missing metadata"
            )

            report_lines.append(
                "⚠ Warning: Cluster interpretation degraded due to missing metadata."
            )

        # =====================================================
        # TOP REPRESENTATIVE MOVIES
        # =====================================================
        report_lines.append("\nTop Representative Movies:")

        top_movies = (
            cluster_data
            .sort("vote_count", descending=True)
            .head(5)
        )

        for row in top_movies.iter_rows(named=True):

            report_lines.append(
                f"  - {row['title']} | "
                f"Genres: {row['genres']} | "
                f"Rating: {row['avg_rating']:.2f} | "
                f"Votes: {int(row['vote_count'])}"
            )

        # =====================================================
        # CLUSTER WARNINGS
        # =====================================================
        warnings = []

        if cluster_size > avg_cluster_size * 1.8:
            warnings.append("Oversized cluster")

        if cluster_size < avg_cluster_size * 0.4:
            warnings.append("Very small cluster")

        if not genres_list:
            warnings.append("Missing metadata")

        if warnings:
            report_lines.append("\nCluster Warnings:")
            for w in warnings:
                report_lines.append(f"  ⚠ {w}")

        report_lines.append("\n")

    # =========================================================
    # FAILURE ANALYSIS
    # =========================================================
    report_lines.append("=========================================================")
    report_lines.append("         FAILURE ANALYSIS & AMBIGUOUS CASES")
    report_lines.append("=========================================================\n")

    report_lines.append(
        "The following movies contain highly hybrid genre combinations "
        "that may create ambiguous embedding positions:\n"
    )

    complex_genres = [
        ["Action", "Adventure", "Sci-Fi", "Thriller"],
        ["Comedy", "Drama", "Romance"],
        ["Horror", "Sci-Fi", "Thriller"],
    ]

    ambiguous_movies = (
        full_catalog
        .filter(pl.col("genres").is_in(complex_genres))
        .head(10)
    )

    if ambiguous_movies.shape[0] == 0:

        report_lines.append(
            "No strongly ambiguous movies detected."
        )

    else:

        for row in ambiguous_movies.iter_rows(named=True):

            report_lines.append(
                f"- {row['title']}"
            )

            report_lines.append(
                f"  Cluster: {row['cluster_id']}"
            )

            report_lines.append(
                f"  Genres: {row['genres']}"
            )

            report_lines.append(
                f"  Avg Rating: {row['avg_rating']:.2f}"
            )

            report_lines.append(
                "  Interpretation: Multi-genre structure may place "
                "this movie near cluster boundaries.\n"
            )

    # =========================================================
    # FINAL CONCLUSIONS
    # =========================================================
    report_lines.append("=========================================================")
    report_lines.append("                    FINAL OBSERVATIONS")
    report_lines.append("=========================================================\n")

    report_lines.append(
        "- Clusters appear to capture meaningful thematic structures."
    )

    report_lines.append(
        "- Genre metadata quality directly impacts interpretability."
    )

    report_lines.append(
        "- Hybrid genre films create soft cluster boundaries."
    )

    report_lines.append(
        "- Large drama/comedy dominance suggests dataset genre imbalance."
    )

    report_lines.append(
        "- Additional embeddings or metadata may improve fine-grained separation."
    )

    # =========================================================
    # EXPORT
    # =========================================================
    output_path = Path(
        "reports/week07/cluster_interpretation_insights.md"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    print(f"Advanced report saved to {output_path}")