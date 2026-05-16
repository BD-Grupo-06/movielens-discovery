from sklearn.preprocessing import StandardScaler
import polars as pl
from sentence_transformers import SentenceTransformer
import numpy as np

def build_genre_features(
    movies_df: pl.DataFrame
) -> pl.DataFrame:
    genre_dummies = (
        movies_df
        .select(["movieId", "genres"])
        .explode("genres")
        .rename({"genres": "genre"})
        .to_dummies(columns=["genre"])
        .group_by("movieId")
        .sum()
    )
    return genre_dummies
def build_rating_features(ratings_df: pl.DataFrame) -> pl.DataFrame:
    if "timestamp" in ratings_df.columns:
        ratings_df = ratings_df.with_columns(pl.col("timestamp").cast(pl.Int64))

    agg = ratings_df.group_by("movieId").agg(
        [
            pl.count().alias("rating_count"),
            pl.col("rating").mean().alias("rating_mean"),
        ]
    )

    agg = agg.with_columns(
        pl.col("rating_count").log1p().alias("rating_count_log")
    )

    scaler = StandardScaler()
    
    scaled_matrix = scaler.fit_transform(
        agg.select(["rating_count_log", "rating_mean"]).to_numpy()
    )

    agg = agg.with_columns([
        pl.Series("rating_count", scaled_matrix[:, 0]),
        pl.Series("rating_mean", scaled_matrix[:, 1])
    ]).drop("rating_count_log")

    return agg

def build_tag_text(movies_df: pl.DataFrame, tags_df: pl.DataFrame) -> pl.DataFrame:

    movie_tags = (
        tags_df
        .group_by("movieId")
        .agg([
            pl.col("tag")
            .drop_nulls()
            .unique()
            .alias("tags_list")
        ])
        .with_columns(
            pl.col("tags_list")
            .list.join(" ")
            .alias("tags_text")
        )
    )

    df = (
        movies_df
        .select(["movieId", "title"])
        .join(movie_tags, on="movieId", how="left")
        .with_columns([
            pl.col("title").str.to_lowercase(),
            pl.col("tags_text").fill_null("")
        ])
        .with_columns(
            (pl.col("title") + " " + pl.col("tags_text"))
            .str.strip_chars()
            .alias("movie_context")
        )
        .select(["movieId", "movie_context"])
    ).sort("movieId")

    return df


def run_embeddings(corpus: list[str], model_name: str = "all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        corpus,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return np.array(embeddings), model
