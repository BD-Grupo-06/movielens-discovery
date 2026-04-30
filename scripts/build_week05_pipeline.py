from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional
import logging

DEFAULT_RANDOM_STATE = 42


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_parquet(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pl.read_parquet(path)


def build_genre_features(
    movies_df: pl.DataFrame, movie_genres_df: pl.DataFrame
) -> pl.DataFrame:
    if "genre" in movie_genres_df.columns:
        base = movie_genres_df.select(["movieId", "genre"])
    elif "genres" in movie_genres_df.columns:
        base = (
            movie_genres_df.select(["movieId", "genres"])
            .with_columns(
                pl.col("genres")
                .fill_null("")
                .str.split("|")
                .alias("genre")
            )
            .explode("genre")
            .filter(pl.col("genre") != "")
            .select(["movieId", "genre"])
        )
    elif "genres" in movies_df.columns:
        base = (
            movies_df.select(["movieId", "genres"])
            .with_columns(
                pl.col("genres")
                .fill_null("")
                .str.split("|")
                .alias("genre")
            )
            .explode("genre")
            .filter(pl.col("genre") != "")
            .select(["movieId", "genre"])
        )
    else:
        return movies_df.select("movieId")

    genre_dummies = base.to_dummies(columns=["genre"]).group_by("movieId").sum()
    return genre_dummies


def build_rating_features(ratings_df: pl.DataFrame) -> pl.DataFrame:
    if "timestamp" in ratings_df.columns:
        ratings_df = ratings_df.with_columns(pl.col("timestamp").cast(pl.Int64))

    agg = ratings_df.group_by("movieId").agg(
        [
            pl.len().alias("rating_count"),
            pl.col("rating").mean().alias("rating_mean"),
            pl.col("rating").std().alias("rating_std"),
            pl.col("rating").min().alias("rating_min"),
            pl.col("rating").max().alias("rating_max"),
            pl.col("rating").median().alias("rating_median"),
            pl.col("timestamp").min().alias("rating_first_ts"),
            pl.col("timestamp").max().alias("rating_last_ts"),
        ]
    )

    if "rating_first_ts" in agg.columns and "rating_last_ts" in agg.columns:
        agg = agg.with_columns(
            ((pl.col("rating_last_ts") - pl.col("rating_first_ts")) / 86400.0).alias(
                "rating_span_days"
            )
        )

    return agg


def build_tag_text(tags_df: pl.DataFrame) -> pl.DataFrame:
    tag_col = None
    for candidate in ["tag", "tags", "tag_text"]:
        if candidate in tags_df.columns:
            tag_col = candidate
            break

    if tag_col is None:
        return pl.DataFrame({"movieId": [], "tag_text": [], "tag_count": []})

    return (
        tags_df.select(["movieId", tag_col])
        .with_columns(pl.col(tag_col).cast(pl.Utf8))
        .group_by("movieId")
        .agg(
            [
                pl.col(tag_col).unique().alias("tag_list"),
                pl.len().alias("tag_count"),
            ]
        )
        .with_columns(pl.col("tag_list").list.join(" ").alias("tag_text"))
        .select(["movieId", "tag_text", "tag_count"])
    )


def build_text_corpus(
    movies_df: pl.DataFrame, tags_agg: pl.DataFrame
) -> pl.DataFrame:
    text_df = movies_df.select(["movieId", "title"]).with_columns(
        pl.col("title").cast(pl.Utf8).fill_null("")
    )

    if tags_agg.height > 0:
        text_df = text_df.join(tags_agg, on="movieId", how="left")
    else:
        text_df = text_df.with_columns(pl.lit("").alias("tag_text"))

    text_df = text_df.with_columns(
        (
            pl.col("title").fill_null("")
            + pl.lit(" ")
            + pl.col("tag_text").fill_null("")
        ).alias("text_blob")
    )

    return text_df.select(["movieId", "text_blob", "tag_count"])


def build_feature_matrices(
    movies_df: pl.DataFrame,
    movie_genres_df: pl.DataFrame,
    ratings_df: pl.DataFrame,
    tags_df: pl.DataFrame,
    max_features: int = 5000,
) -> Tuple[pd.DataFrame, sparse.csr_matrix, Dict[str, int]]:
    genre_features = build_genre_features(movies_df, movie_genres_df)
    rating_features = build_rating_features(ratings_df)
    tag_agg = build_tag_text(tags_df)

    numeric_features = rating_features.join(genre_features, on="movieId", how="left")

    if "tag_count" in tag_agg.columns:
        numeric_features = numeric_features.join(
            tag_agg.select(["movieId", "tag_count"]), on="movieId", how="left"
        )

    numeric_features = numeric_features.fill_null(0).sort("movieId")

    text_df = build_text_corpus(movies_df, tag_agg).sort("movieId")
    corpus = text_df.get_column("text_blob").to_list()

    vectorizer = TfidfVectorizer(
        max_features=max_features, min_df=2, stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    return (
        numeric_features.to_pandas(),
        tfidf_matrix,
        vectorizer.vocabulary_,
    )


def run_pca(numeric_df: pd.DataFrame, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    movie_ids = numeric_df["movieId"].values
    numeric_only = numeric_df.drop(columns=["movieId"])
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_only)

    pca = PCA(n_components=0.9, random_state=random_state)
    pca_result = pca.fit_transform(numeric_scaled)

    pca_df = pd.DataFrame(pca_result)
    pca_df.insert(0, "movieId", movie_ids)

    variance_df = pd.DataFrame(
        {
            "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    return pca_df, variance_df


def run_svd(tfidf_matrix: sparse.csr_matrix, random_state: int) -> Tuple[np.ndarray, pd.DataFrame]:
    n_components = min(200, tfidf_matrix.shape[1] - 1) if tfidf_matrix.shape[1] > 1 else 1
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    svd_result = svd.fit_transform(tfidf_matrix)

    variance_df = pd.DataFrame(
        {
            "component": np.arange(1, len(svd.explained_variance_ratio_) + 1),
            "explained_variance_ratio": svd.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(svd.explained_variance_ratio_),
        }
    )

    return svd_result, variance_df


def reconstruction_error_pca(numeric_df: pd.DataFrame, random_state: int) -> float:
    numeric_only = numeric_df.drop(columns=["movieId"])
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_only)

    pca = PCA(n_components=0.9, random_state=random_state)
    transformed = pca.fit_transform(numeric_scaled)
    reconstructed = pca.inverse_transform(transformed)

    error = np.mean((numeric_scaled - reconstructed) ** 2)
    return float(error)


def reconstruction_error_svd(tfidf_matrix: sparse.csr_matrix, random_state: int) -> float:
    n_components = min(200, tfidf_matrix.shape[1] - 1) if tfidf_matrix.shape[1] > 1 else 1
    
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    svd.fit(tfidf_matrix)

    explained = svd.explained_variance_ratio_.sum()
    error = 1.0 - explained

    return float(error)


def run_tsne(embedding: np.ndarray, random_state: int) -> np.ndarray:
    tsne = TSNE(n_components=2, random_state=random_state, init="random", perplexity=30)
    return tsne.fit_transform(embedding)


def save_plot(fig, path: Path) -> None:
    fig.write_html(str(path))
    png_path = path.with_suffix(".png")
    try:
        fig.write_image(str(png_path))
    except Exception:
        pass


def run_eda(input_dir: Path | str, output_dir: Path | str) -> None:
    """
    Performs Expanded Exploratory Data Analysis on the Silver/processed layer data.
    Saves several PNG visualizations to the output directory.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info("Starting Expanded Exploratory Data Analysis")

    # Locate movies file (check common parquet/csv locations)
    candidates_movies = [
        input_dir / "movies_catalog.parquet",
        input_dir / "movies.parquet",
        input_dir / "movies.csv",
        Path("data/raw/ml-25m/movies.csv"),
    ]
    movies = None
    movies_path_used = None
    for p in candidates_movies:
        if p.exists():
            movies_path_used = p
            logger.info("Loading movies from %s", p)
            if p.suffix == ".parquet":
                movies = pl.read_parquet(p)
            else:
                movies = pl.read_csv(p)
            break

    if movies is None:
        logger.warning("No movies file found in candidates; skipping EDA.")
        return

    # Locate movie stats (prefer processed numeric features), else derive from ratings
    candidates_stats = [
        input_dir / "movie_numeric_features.parquet",
        Path("data/processed/week05/movie_numeric_features.parquet"),
        Path("data/processed/week03_v1/movie_numeric_features.parquet"),
    ]
    movie_stats = None
    stats_path_used = None
    for p in candidates_stats:
        if p.exists():
            stats_path_used = p
            logger.info("Loading movie numeric stats from %s", p)
            movie_stats = pl.read_parquet(p)
            break

    if movie_stats is None:
        # try to find ratings to aggregate
        candidates_ratings = [
            input_dir / "ratings_clean.parquet",
            Path("data/processed/week03_v1/ratings_clean.parquet"),
            Path("data/raw/ml-25m/ratings.csv"),
        ]
        ratings = None
        ratings_path_used = None
        for p in candidates_ratings:
            if p.exists():
                ratings_path_used = p
                logger.info("Loading ratings from %s to derive stats", p)
                if p.suffix == ".parquet":
                    ratings = pl.read_parquet(p)
                else:
                    ratings = pl.read_csv(p)
                break

        if ratings is not None:
            movie_stats = (
                ratings.group_by("movieId").agg([
                    pl.len().alias("rating_count"),
                    pl.col("rating").mean().alias("mean_rating"),
                    pl.col("rating").var().alias("rating_variance"),
                ])
            )
            logger.info("Derived movie stats from ratings (rows=%d)", movie_stats.height)
        else:
            logger.warning("No movie numeric stats or ratings found; proceeding with limited EDA (movies only)")
            movie_stats = pl.DataFrame({"movieId": []})

    logger.debug("Merging data for EDA...")
    # Convert to Pandas for visualization
    # Ensure consistent column names
    stats = movie_stats
    # normalise column names used below
    if "rating_count" in stats.columns and "mean_rating" not in stats.columns:
        if "rating_mean" in stats.columns:
            stats = stats.rename({"rating_mean": "mean_rating"})

    stats = movie_stats
    # normalize rating column names
    if "rating_mean" in stats.columns and "mean_rating" not in stats.columns:
        stats = stats.rename({"rating_mean": "mean_rating"})

    # join (left join so movies without stats are kept)
    df = movies.join(stats, on="movieId", how="left").to_pandas()

    # 2. Genre Volume Analysis
    logger.info("Analyzing Genre Volume...")
    if "genres" in movies.columns:
        genre_counts = (
            movies.select(pl.col("genres").str.split("|").alias("genres"))
            .explode("genres")
            .with_columns(pl.col("genres").alias("genre"))
            .group_by("genres")
            .count()
            .sort("count", descending=True)
        ).to_pandas()

        plt.figure(figsize=(12, 6))
        sns.barplot(data=genre_counts, x="count", y="genres", palette="viridis")
        plt.title("Number of Movies per Genre")
        plt.xlabel("Movie Count")
        plt.ylabel("Genre")
        plt.tight_layout()
        plt.savefig(output_dir / "genre_volume.png")
        plt.close()
    else:
        logger.info("No genres column in movies dataset; skipping genre volume.")

    # 3. Genre Quality Profiling: Mean Rating Distribution by Genre
    logger.info("Profiling Genre Quality (Mean Ratings)...")
    if "genres" in movies.columns and "mean_rating" in df.columns:
        genre_quality = (
            movies.join(stats, on="movieId", how="inner")
            .select([pl.col("genres").str.split("|").alias("genres"), "mean_rating"]) 
            .explode("genres")
        ).to_pandas()

        plt.figure(figsize=(12, 8))
        order = genre_quality.groupby("genres")["mean_rating"].median().sort_values(ascending=False).index
        sns.boxplot(data=genre_quality, x="mean_rating", y="genres", order=order, palette="coolwarm")
        plt.title("Distribution of Mean Ratings by Genre (Ordered by Median)")
        plt.xlabel("Mean Rating")
        plt.ylabel("Genre")
        plt.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "genre_quality_profile.png")
        plt.close()
    else:
        logger.info("Insufficient data for genre quality profiling; skipping.")

    # 4. Popularity vs Quality Correlation
    logger.info("Analyzing Popularity vs Quality...")
    if "rating_count" in df.columns and "mean_rating" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="rating_count", y="mean_rating", alpha=0.1, color="purple")
        plt.xscale("log")
        plt.title("Popularity (Count) vs Quality (Mean Rating)")
        plt.xlabel("Rating Count (Log Scale)")
        plt.ylabel("Mean Rating")
        plt.tight_layout()
        plt.savefig(output_dir / "popularity_vs_quality.png")
        plt.close()
    else:
        logger.info("Insufficient columns for popularity vs quality plot; skipping.")

    # 5. Long-Tail Plot
    logger.info("Generating Long-Tail Plot...")
    if "rating_count" in df.columns:
        popularity = df["rating_count"].sort_values(ascending=False).values
        plt.figure(figsize=(10, 6))
        plt.plot(popularity, color="darkorange", linewidth=2)
        plt.title("Long-Tail Distribution of Movie Popularity")
        plt.xlabel("Movie Rank")
        plt.ylabel("Number of Ratings")
        plt.yscale("log")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "long_tail.png")
        plt.close()
    else:
        logger.info("No rating_count column for long-tail plot; skipping.")
    logger.info("Expanded EDA plots saved to %s", output_dir)


def run_pipeline(
    data_dir: Path,
    processed_dir: Path,
    artifacts_dir: Path,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Starting pipeline: data_dir=%s processed_dir=%s artifacts_dir=%s", data_dir, processed_dir, artifacts_dir)
    import time

    start_total = time.perf_counter()

    logger.info("Ensuring directories...")
    ensure_dir(processed_dir)
    ensure_dir(artifacts_dir)

    # Always run EDA before processing
    try:
        logger.info("Running pre-processing EDA...")
        run_eda(input_dir=data_dir, output_dir=artifacts_dir)
        logger.info("Completed pre-processing EDA")
    except Exception as e:
        logger.warning("Pre-processing EDA failed: %s", e)

    t0 = time.perf_counter()
    logger.info("Reading input parquet files...")
    movies_df = read_parquet(data_dir / "movies_catalog.parquet")
    movie_genres_df = read_parquet(data_dir / "movie_genres.parquet")
    ratings_df = read_parquet(data_dir / "ratings_clean.parquet")
    tags_df = read_parquet(data_dir / "tags_clean.parquet")
    logger.info("Read input files in %.2fs", time.perf_counter() - t0)

    t0 = time.perf_counter()
    logger.info("Building feature matrices (genre, rating, text)...")
    numeric_df, tfidf_matrix, vocab = build_feature_matrices(
        movies_df, movie_genres_df, ratings_df, tags_df
    )
    logger.info("Built feature matrices in %.2fs", time.perf_counter() - t0)

    numeric_path = processed_dir / "movie_numeric_features.parquet"
    t0 = time.perf_counter()
    logger.info("Writing numeric features to %s", numeric_path)
    pl.from_pandas(numeric_df).write_parquet(numeric_path)
    logger.info("Wrote numeric features in %.2fs", time.perf_counter() - t0)

    tfidf_path = processed_dir / "movie_text_tfidf.npz"
    t0 = time.perf_counter()
    logger.info("Saving TF-IDF matrix to %s", tfidf_path)
    sparse.save_npz(str(tfidf_path), tfidf_matrix)
    logger.info("Saved TF-IDF in %.2fs", time.perf_counter() - t0)

    vocab_path = processed_dir / "movie_text_tfidf_vocab.json"
    vocab_serializable = {key: int(value) for key, value in vocab.items()}
    logger.info("Writing vocab to %s", vocab_path)
    vocab_path.write_text(json.dumps(vocab_serializable, indent=2, ensure_ascii=True))

    t0 = time.perf_counter()
    logger.info("Running PCA and SVD (this may take a while)...")
    pca_df, pca_variance = run_pca(numeric_df, random_state)
    svd_embeddings, svd_variance = run_svd(tfidf_matrix, random_state)
    logger.info("Completed PCA/SVD in %.2fs", time.perf_counter() - t0)

    logger.info("Writing PCA embeddings to parquet")
    pl.from_pandas(pca_df).write_parquet(
        processed_dir / "movie_pca_embeddings.parquet"
    )

    svd_columns = [f"svd_{idx + 1}" for idx in range(svd_embeddings.shape[1])]
    logger.info("Writing SVD embeddings to parquet")
    pl.DataFrame(svd_embeddings, schema=svd_columns).write_parquet(
        processed_dir / "movie_svd_embeddings.parquet"
    )

    logger.info("Writing variance CSVs to artifacts")
    pca_variance.to_csv(artifacts_dir / "pca_variance.csv", index=False)
    svd_variance.to_csv(artifacts_dir / "svd_variance.csv", index=False)

    t0 = time.perf_counter()
    logger.info("Computing reconstruction errors...")
    pca_error = reconstruction_error_pca(numeric_df, random_state)
    svd_error = reconstruction_error_svd(tfidf_matrix, random_state)
    logger.info("Computed reconstruction errors in %.2fs", time.perf_counter() - t0)

    comparison_table = pd.DataFrame(
        [
            {
                "method": "PCA",
                "components": int(pca_variance["component"].max()),
                "cumulative_variance": float(pca_variance["cumulative_variance"].max()),
                "reconstruction_error": pca_error,
            },
            {
                "method": "TruncatedSVD",
                "components": int(svd_variance["component"].max()),
                "cumulative_variance": float(svd_variance["cumulative_variance"].max()),
                "reconstruction_error": svd_error,
            },
        ]
    )

    logger.info("Writing comparison table to artifacts")
    comparison_table.to_csv(artifacts_dir / "comparison_table.csv", index=False)

    logger.info("Running t-SNE on PCA embeddings")
    tsne_input = pca_df.drop(columns=["movieId"]).values
    tsne_coords = run_tsne(tsne_input, random_state)
    tsne_df = pd.DataFrame(tsne_coords, columns=["tsne_1", "tsne_2"])
    tsne_df.insert(0, "movieId", pca_df["movieId"].values)
    tsne_df.to_parquet(artifacts_dir / "pca_tsne.parquet", index=False)

    pca_fig = px.line(
        pca_variance,
        x="component",
        y="cumulative_variance",
        markers=True,
        title="PCA Cumulative Explained Variance (Numeric Features)",
    )
    save_plot(pca_fig, artifacts_dir / "pca_cumulative_variance.html")
    logger.info("Saved PCA cumulative variance plots")

    svd_fig = px.line(
        svd_variance,
        x="component",
        y="cumulative_variance",
        markers=True,
        title="SVD Cumulative Explained Variance (Text TF-IDF)",
    )
    save_plot(svd_fig, artifacts_dir / "svd_cumulative_variance.html")
    logger.info("Saved SVD cumulative variance plots")

    tsne_fig = px.scatter(
        tsne_df,
        x="tsne_1",
        y="tsne_2",
        title="t-SNE on PCA Embeddings",
        opacity=0.6,
    )
    save_plot(tsne_fig, artifacts_dir / "tsne_scatter.html")
    logger.info("Saved t-SNE scatter plot")

    summary = {
        "numeric_features": {
            "rows": int(numeric_df.shape[0]),
            "columns": int(numeric_df.shape[1]),
        },
        "text_features": {
            "rows": int(tfidf_matrix.shape[0]),
            "columns": int(tfidf_matrix.shape[1]),
        },
        "pca": {
            "components": int(pca_variance.shape[0]),
            "cumulative_variance": float(pca_variance["cumulative_variance"].max()),
        },
        "svd": {
            "components": int(svd_variance.shape[0]),
            "cumulative_variance": float(svd_variance["cumulative_variance"].max()),
        },
        "reconstruction_error": {
            "pca": pca_error,
            "svd": svd_error,
        },
    }

    (artifacts_dir / "week05_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote week05_summary.json and artifacts (pca/svd/comparison). Total time: %.2fs", time.perf_counter() - start_total)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Week 5 feature matrices and dimensionality outputs."
    )
    parser.add_argument(
        "--input-dir",
        default="data/processed/week03_v1",
        help="Path to Week 3 processed data directory.",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed/week05",
        help="Path to write Week 5 processed outputs.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/week05",
        help="Path to write Week 5 artifacts.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for pipeline output.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for PCA/SVD/t-SNE.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    run_pipeline(
        data_dir=Path(args.input_dir),
        processed_dir=Path(args.processed_dir),
        artifacts_dir=Path(args.artifacts_dir),
        random_state=args.random_state,
        
    )


if __name__ == "__main__":
    main()
