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
            pl.count().alias("rating_count"),
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
                pl.count().alias("tag_count"),
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
    transformed = svd.fit_transform(tfidf_matrix)
    reconstructed = svd.inverse_transform(transformed)
    diff = tfidf_matrix - sparse.csr_matrix(reconstructed)
    return float((diff.multiply(diff)).sum() / tfidf_matrix.shape[0])


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


def run_pipeline(
    data_dir: Path,
    processed_dir: Path,
    artifacts_dir: Path,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> None:
    ensure_dir(processed_dir)
    ensure_dir(artifacts_dir)

    movies_df = read_parquet(data_dir / "movies_catalog.parquet")
    movie_genres_df = read_parquet(data_dir / "movie_genres.parquet")
    ratings_df = read_parquet(data_dir / "ratings_clean.parquet")
    tags_df = read_parquet(data_dir / "tags_clean.parquet")

    numeric_df, tfidf_matrix, vocab = build_feature_matrices(
        movies_df, movie_genres_df, ratings_df, tags_df
    )

    numeric_path = processed_dir / "movie_numeric_features.parquet"
    pl.from_pandas(numeric_df).write_parquet(numeric_path)

    tfidf_path = processed_dir / "movie_text_tfidf.npz"
    sparse.save_npz(str(tfidf_path), tfidf_matrix)

    vocab_path = processed_dir / "movie_text_tfidf_vocab.json"
    vocab_serializable = {key: int(value) for key, value in vocab.items()}
    vocab_path.write_text(json.dumps(vocab_serializable, indent=2, ensure_ascii=True))

    pca_df, pca_variance = run_pca(numeric_df, random_state)
    svd_embeddings, svd_variance = run_svd(tfidf_matrix, random_state)

    pl.from_pandas(pca_df).write_parquet(
        processed_dir / "movie_pca_embeddings.parquet"
    )

    svd_columns = [f"svd_{idx + 1}" for idx in range(svd_embeddings.shape[1])]
    pl.DataFrame(svd_embeddings, schema=svd_columns).write_parquet(
        processed_dir / "movie_svd_embeddings.parquet"
    )

    pca_variance.to_csv(artifacts_dir / "pca_variance.csv", index=False)
    svd_variance.to_csv(artifacts_dir / "svd_variance.csv", index=False)

    pca_error = reconstruction_error_pca(numeric_df, random_state)
    svd_error = reconstruction_error_svd(tfidf_matrix, random_state)

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

    comparison_table.to_csv(artifacts_dir / "comparison_table.csv", index=False)

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

    svd_fig = px.line(
        svd_variance,
        x="component",
        y="cumulative_variance",
        markers=True,
        title="SVD Cumulative Explained Variance (Text TF-IDF)",
    )
    save_plot(svd_fig, artifacts_dir / "svd_cumulative_variance.html")

    tsne_fig = px.scatter(
        tsne_df,
        x="tsne_1",
        y="tsne_2",
        title="t-SNE on PCA Embeddings",
        opacity=0.6,
    )
    save_plot(tsne_fig, artifacts_dir / "tsne_scatter.html")

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
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for PCA/SVD/t-SNE.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        data_dir=Path(args.input_dir),
        processed_dir=Path(args.processed_dir),
        artifacts_dir=Path(args.artifacts_dir),
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
