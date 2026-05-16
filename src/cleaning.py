import polars as pl
from pathlib import Path
from src.config import RAW_FILES, DATASET_DIR_NAME

def table_path(raw_dir: Path, table_name: str) -> Path:
    return raw_dir / DATASET_DIR_NAME / RAW_FILES[table_name]

def to_int64_nullable(column_name: str) -> pl.Expr:
    return pl.col(column_name).cast(pl.Int64, strict=False)

def to_float64(column_name: str) -> pl.Expr:
    return pl.col(column_name).cast(pl.Float64, strict=False)

def clean_movies(raw_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    movies_raw = pl.read_csv(table_path(raw_dir, "movies"), infer_schema_length=10_000)
    links_raw = pl.read_csv(table_path(raw_dir, "links"), infer_schema_length=10_000)

    movies_clean = (
        movies_raw.select(["movieId", "title", "genres"])
        .with_columns(
            [
                to_int64_nullable("movieId"),
                pl.col("title").cast(pl.String, strict=False),
                pl.col("genres").cast(pl.String, strict=False),
            ]
        )
        .filter(pl.col("movieId").is_not_null() & pl.col("title").is_not_null())
        .join(
            links_raw.select(["movieId", "imdbId", "tmdbId"]).with_columns(
                [
                    to_int64_nullable("movieId"),
                    to_int64_nullable("imdbId"),
                    to_int64_nullable("tmdbId"),
                ]
            ),
            on="movieId",
            how="left",
        )
        .with_columns(
            [
                pl.col("title").str.extract(r"\((\d{4})\)$", 1).cast(pl.Int64, strict=False).alias("release_year"),
                pl.col("title").str.replace(r"\s*\(\d{4}\)$", "").alias("title"),
                pl.when(pl.col("genres") == "(no genres listed)").then(None).otherwise(pl.col("genres")).str.split("|").alias("genres"),
            ]
        )
        .with_columns(
            pl.when(pl.col("imdbId").is_not_null()).then(pl.format("tt{}", pl.col("imdbId"))).otherwise(None).alias("imdb_title_id"),
        )
        .sort("movieId")
    )
    return movies_raw, links_raw, movies_clean

def clean_ratings(raw_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    ratings_raw = pl.read_csv(table_path(raw_dir, "ratings"), infer_schema_length=10_000)
    ratings_clean = (
        ratings_raw.select(["userId", "movieId", "rating", "timestamp"])
        .with_columns(
            [
                to_int64_nullable("userId"),
                to_int64_nullable("movieId"),
                to_float64("rating"),
                to_int64_nullable("timestamp"),
            ]
        )
        .filter(
            pl.col("userId").is_not_null()
            & pl.col("movieId").is_not_null()
            & pl.col("rating").is_not_null()
            & pl.col("timestamp").is_not_null()
        )
        .filter((pl.col("rating") >= 0.5) & (pl.col("rating") <= 5.0))
        .with_columns(pl.from_epoch("timestamp", time_unit="s").alias("rated_at"))
        .sort(["userId", "movieId", "timestamp"])
    )
    return ratings_raw, ratings_clean

def clean_tags(raw_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    tags_raw = pl.read_csv(
        table_path(raw_dir, "tags"),
        infer_schema_length=10_000
    )

    tags_clean = (
        tags_raw
        .select(["userId", "movieId", "tag", "timestamp"])
        .with_columns([
            to_int64_nullable("userId"),
            to_int64_nullable("movieId"),
            pl.col("tag").cast(pl.String, strict=False),
            to_int64_nullable("timestamp"),
        ])
        .filter(
            pl.col("userId").is_not_null() &
            pl.col("movieId").is_not_null() &
            pl.col("tag").is_not_null() &
            pl.col("timestamp").is_not_null()
        )
        .with_columns(
            pl.col("tag")
            .str.strip_chars()
            .str.to_lowercase()
            .str.replace_all(r"[^\w\s]", "")
            .alias("tag")
        )
        .filter(pl.col("tag").str.len_chars() > 0)
        .unique(subset=["userId", "movieId", "tag"])
    )

    return tags_raw, tags_clean

def build_movie_genres(movies_clean: pl.DataFrame) -> pl.DataFrame:
    return (
        movies_clean.select(["movieId", "genres"])
        .explode("genres")
        .rename({"genres": "genre"})
        .drop_nulls(["genre"])
        .sort(["movieId", "genre"])
    )
