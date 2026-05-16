import pandas as pd
import json
from pathlib import Path
import polars as pl
from scipy import sparse
from src.config import DEFAULT_RANDOM_STATE
from src.io import ensure_dir, read_parquet
from src.features.content import build_genre_features, build_tag_text, run_embeddings, build_rating_features
from src.features.collaborative import run_svd_collaborative
from src.models.decomposition import run_tsne, run_pca_embeddings
from src.models.reconstruction import reconstruction_error_pca, reconstruction_error_svd
from src.visualization import save_plot, plot_cumulative_variance, plot_tsne_scatter

def run_week05_pipeline(
    data_dir: Path,
    processed_dir: Path,
    artifacts_dir: Path,
    random_state: int = DEFAULT_RANDOM_STATE
):
    ensure_dir(processed_dir)
    ensure_dir(artifacts_dir)

    print("Loading data...")
    movies_df = read_parquet(data_dir / "movies_catalog.parquet")
    ratings_df = read_parquet(data_dir / "ratings_clean.parquet")
    tags_df = read_parquet(data_dir / "tags_clean.parquet")

    print("Building content features...")
    genre_features = build_genre_features(movies_df)
    rating_features = build_rating_features(ratings_df)
    movie_context = build_tag_text(movies_df, tags_df)

    # Text embeddings
    corpus = movie_context.get_column("movie_context").to_list()
    embeddings, model = run_embeddings(corpus)

    movie_ids = movie_context.get_column("movieId").to_list()

    pca_embeddings, pca_variance, pca_model = run_pca_embeddings(
        embeddings,
        movie_ids,
        n_components=0.9,
        random_state=random_state
    )
    
    movie_features = (
        genre_features
        .join(rating_features, on="movieId", how="left")
        .join(pca_embeddings, on="movieId", how="left")
        .fill_null(0)
        .sort("movieId")
    )

    movie_features.write_parquet(processed_dir / "movie_features.parquet")

    print("Running SVD collaborative filtering...")
    user_df, item_df, svd_model = run_svd_collaborative(
        ratings_df,
        n_components=50,
        random_state=random_state
    )

    user_df.write_parquet(processed_dir / "user_embeddings.parquet")
    item_df.write_parquet(processed_dir / "item_embeddings.parquet")

    print("Calculating reconstruction errors...")
    pca_error = reconstruction_error_pca(embeddings, pca_model)
    # svd_error = reconstruction_error_svd(ratings_df, user_df, item_df)

    print("Running t-SNE for visualization...")
    movie_features_pd = movie_features.to_pandas()
    
    tsne_input = movie_features_pd.drop(columns=["movieId"]).values
    
    tsne_coords, indices = run_tsne(tsne_input, random_state)
    
    tsne_df = pd.DataFrame(tsne_coords, columns=["tsne_1", "tsne_2"])
    tsne_df.insert(0, "movieId", movie_features_pd["movieId"].values[indices])
    tsne_df.to_parquet(artifacts_dir / "pca_tsne.parquet", index=False)

    print("Generating plots...")
    pca_fig = plot_cumulative_variance(pca_variance, "PCA Cumulative Explained Variance (Text Embeddings)")
    save_plot(pca_fig, artifacts_dir / "pca_cumulative_variance.html")

    tsne_fig = plot_tsne_scatter(tsne_df, "t-SNE on PCA Embeddings")
    save_plot(tsne_fig, artifacts_dir / "tsne_scatter.html")

    # Summary
    summary = {
        "pca": {"components": int(pca_variance.shape[0]), "error": pca_error},
        # "svd": {"components": 50, "error": svd_error},
    }
    (artifacts_dir / "week05_summary.json").write_text(json.dumps(summary, indent=2))
    print("Week 05 pipeline completed successfully.")
