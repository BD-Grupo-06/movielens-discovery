#!/usr/bin/env python3
"""End-to-end Week 3 pipeline for MovieLens 25M."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import polars as pl

DOWNLOAD_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
DATASET_DIR_NAME = "ml-25m"
ARCHIVE_NAME = "ml-25m.zip"
RAW_FILES = {
    "movies": "movies.csv",
    "ratings": "ratings.csv",
    "tags": "tags.csv",
    "links": "links.csv",
    "genome_scores": "genome-scores.csv",
    "genome_tags": "genome-tags.csv",
}
CORE_TABLES = ["movies", "ratings", "tags", "links"]
ALL_TABLES = list(RAW_FILES.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download, extract, clean, and profile MovieLens 25M.")
    parser.add_argument("--download-url", default=DOWNLOAD_URL)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/week03_v1"))
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def dataset_root(raw_dir: Path) -> Path:
    return raw_dir / DATASET_DIR_NAME


def table_path(raw_dir: Path, table_name: str) -> Path:
    return dataset_root(raw_dir) / RAW_FILES[table_name]


def ensure_directories(raw_dir: Path, processed_dir: Path, interim_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)


def dataset_available(raw_dir: Path) -> bool:
    return all(table_path(raw_dir, table_name).exists() for table_name in CORE_TABLES)


def download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as target:
        shutil.copyfileobj(response, target)


def safe_extract(zip_path: Path, destination_dir: Path) -> None:
    destination_dir = destination_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            resolved = (destination_dir / member.filename).resolve()
            if destination_dir not in resolved.parents and resolved != destination_dir:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        archive.extractall(destination_dir)


def ensure_dataset(raw_dir: Path, download_url: str, force_download: bool, skip_download: bool, keep_archive: bool) -> None:
    if dataset_available(raw_dir) and not force_download:
        return

    if skip_download:
        missing = [str(table_path(raw_dir, table_name)) for table_name in CORE_TABLES if not table_path(raw_dir, table_name).exists()]
        raise FileNotFoundError("Missing raw files and downloads are disabled: " + ", ".join(missing))

    archive_path = raw_dir / ARCHIVE_NAME
    if archive_path.exists() and force_download:
        archive_path.unlink()

    print(f"Downloading {download_url} -> {archive_path}")
    download_file(download_url, archive_path)

    print(f"Extracting {archive_path} -> {raw_dir}")
    safe_extract(archive_path, raw_dir)

    if not keep_archive:
        archive_path.unlink(missing_ok=True)

    if not dataset_available(raw_dir):
        raise FileNotFoundError("Extraction finished, but expected MovieLens files are still missing.")


def to_int64_nullable(column_name: str) -> pl.Expr:
    return pl.col(column_name).cast(pl.Int64, strict=False)


def to_float64(column_name: str) -> pl.Expr:
    return pl.col(column_name).cast(pl.Float64, strict=False)


def profile_columns(df: pl.DataFrame, table_name: str) -> pl.DataFrame:
    rows = df.height
    records = []
    for column_name, dtype in df.schema.items():
        null_count = int(df.select(pl.col(column_name).is_null().sum()).item())
        records.append(
            {
                "table": table_name,
                "column": column_name,
                "dtype": str(dtype),
                "null_count": null_count,
                "null_pct": round((null_count / rows) * 100, 4) if rows else 0.0,
            }
        )
    return pl.DataFrame(records)


def build_raw_inventory(raw_dir: Path) -> pl.DataFrame:
    rows = []
    for table_name in ALL_TABLES:
        path = table_path(raw_dir, table_name)
        if path.exists():
            frame = pl.read_csv(path, infer_schema_length=10_000)
            rows.append(
                {
                    "table": table_name,
                    "rows": frame.height,
                    "cols": frame.width,
                    "size_mb": round(path.stat().st_size / (1024**2), 2),
                    "scope": "core" if table_name in CORE_TABLES else "optional",
                }
            )
    return pl.DataFrame(rows)


def build_schema_profile(raw_dir: Path) -> pl.DataFrame:
    rows = []
    for table_name in ALL_TABLES:
        path = table_path(raw_dir, table_name)
        if path.exists():
            frame = pl.read_csv(path, infer_schema_length=10_000)
            for column_name, dtype in frame.schema.items():
                rows.append({"table": table_name, "column": column_name, "dtype": str(dtype)})
    return pl.DataFrame(rows)


def build_missing_profile(raw_dir: Path) -> pl.DataFrame:
    rows = []
    for table_name in ALL_TABLES:
        path = table_path(raw_dir, table_name)
        frame = pl.read_csv(path, infer_schema_length=10_000)
        row_count = frame.height
        for column_name in frame.columns:
            null_count = int(frame.select(pl.col(column_name).is_null().sum()).item())
            rows.append(
                {
                    "table": table_name,
                    "column": column_name,
                    "null_count": null_count,
                    "null_pct": round((null_count / row_count) * 100, 4) if row_count else 0.0,
                }
            )
    return pl.DataFrame(rows)


def build_quality_checks(raw_dir: Path) -> dict[str, int]:
    movies = pl.read_csv(table_path(raw_dir, "movies"), infer_schema_length=10_000)
    ratings = pl.read_csv(table_path(raw_dir, "ratings"), infer_schema_length=10_000)
    tags = pl.read_csv(table_path(raw_dir, "tags"), infer_schema_length=10_000)
    links = pl.read_csv(table_path(raw_dir, "links"), infer_schema_length=10_000)
    genome_scores = pl.read_csv(table_path(raw_dir, "genome_scores"), infer_schema_length=10_000)
    genome_tags = pl.read_csv(table_path(raw_dir, "genome_tags"), infer_schema_length=10_000)

    return {
        "ratings_missing_required": int(
            ratings.select(
                pl.any_horizontal(
                    [
                        pl.col("userId").is_null(),
                        pl.col("movieId").is_null(),
                        pl.col("rating").is_null(),
                        pl.col("timestamp").is_null(),
                    ]
                ).sum()
            ).item()
        ),
        "ratings_out_of_range": int(ratings.filter((pl.col("rating") < 0.5) | (pl.col("rating") > 5.0)).height),
        "ratings_duplicate_user_movie_pairs": int(ratings.select(pl.struct(["userId", "movieId"]).is_duplicated().sum()).item()),
        "movies_duplicate_movieId": int(movies.select(pl.col("movieId").is_duplicated().sum()).item()),
        "movies_missing_title": int(movies.select(pl.col("title").is_null().sum()).item()),
        "movies_missing_genres": int(movies.select(pl.col("genres").is_null().sum()).item()),
        "tags_missing_tag_text": int(tags.select(pl.col("tag").is_null().sum()).item()),
        "tags_duplicate_full_rows": int(tags.is_duplicated().sum()),
        "links_missing_imdbId": int(links.select(pl.col("imdbId").is_null().sum()).item()),
        "links_missing_tmdbId": int(links.select(pl.col("tmdbId").is_null().sum()).item()),
        "links_duplicate_movieId": int(links.select(pl.col("movieId").is_duplicated().sum()).item()),
        "genome_scores_duplicate_movie_tag": int(genome_scores.select(pl.struct(["movieId", "tagId"]).is_duplicated().sum()).item()),
        "genome_tags_duplicate_tagId": int(genome_tags.select(pl.col("tagId").is_duplicated().sum()).item()),
    }


def build_join_coverage(raw_dir: Path) -> pl.DataFrame:
    movies = pl.read_csv(table_path(raw_dir, "movies"), infer_schema_length=10_000)
    ratings = pl.read_csv(table_path(raw_dir, "ratings"), infer_schema_length=10_000)
    tags = pl.read_csv(table_path(raw_dir, "tags"), infer_schema_length=10_000)
    genome_scores = pl.read_csv(table_path(raw_dir, "genome_scores"), infer_schema_length=10_000)
    genome_tags = pl.read_csv(table_path(raw_dir, "genome_tags"), infer_schema_length=10_000)

    movies_ids = movies.select("movieId").unique()
    ratings_ids = ratings.select("movieId").unique()
    tags_ids = tags.select("movieId").unique()
    genome_movie_ids = genome_scores.select("movieId").unique()
    genome_tag_ids = genome_scores.select("tagId").unique()
    genome_tag_lookup = genome_tags.select("tagId").unique()

    orphan_ratings = ratings_ids.join(movies_ids, on="movieId", how="anti").height
    orphan_tags = tags_ids.join(movies_ids, on="movieId", how="anti").height
    orphan_genome_movies = genome_movie_ids.join(movies_ids, on="movieId", how="anti").height
    orphan_genome_tags = genome_tag_ids.join(genome_tag_lookup, on="tagId", how="anti").height

    return pl.DataFrame(
        {
            "relationship_check": [
                "ratings movieId matched in movies",
                "tags movieId matched in movies",
                "genome_scores movieId matched in movies",
                "genome_scores tagId matched in genome_tags",
            ],
            "unmatched_count": [int(orphan_ratings), int(orphan_tags), int(orphan_genome_movies), int(orphan_genome_tags)],
            "coverage_pct": [
                round((1 - (orphan_ratings / max(1, ratings_ids.height))) * 100, 4),
                round((1 - (orphan_tags / max(1, tags_ids.height))) * 100, 4),
                round((1 - (orphan_genome_movies / max(1, genome_movie_ids.height))) * 100, 4),
                round((1 - (orphan_genome_tags / max(1, genome_tag_ids.height))) * 100, 4),
            ],
        }
    )


def build_scale_metrics(raw_dir: Path) -> tuple[dict[str, int | float], pl.DataFrame]:
    ratings = pl.read_csv(table_path(raw_dir, "ratings"), infer_schema_length=10_000)
    users_n = ratings.select(pl.col("userId").n_unique()).item()
    items_n = ratings.select(pl.col("movieId").n_unique()).item()
    interactions_n = ratings.height
    possible_interactions = int(users_n * items_n)
    density = interactions_n / possible_interactions

    metrics = {
        "users": int(users_n),
        "items": int(items_n),
        "interactions": int(interactions_n),
        "possible_user_item_pairs": possible_interactions,
        "density_pct": round(density * 100, 6),
        "sparsity_pct": round((1 - density) * 100, 6),
    }
    return metrics, build_raw_inventory(raw_dir).sort("size_mb", descending=True)


def clean_movies(raw_dir: Path) -> tuple[int, pl.DataFrame, pl.DataFrame]:
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
                [to_int64_nullable("movieId"), to_int64_nullable("imdbId"), to_int64_nullable("tmdbId")]
            ),
            on="movieId",
            how="left",
        )
        .with_columns(
            [
                pl.col("title").str.extract(r"\((\d{4})\)$", 1).cast(pl.Int64, strict=False).alias("release_year"),
                pl.when(pl.col("genres") == "(no genres listed)").then(None).otherwise(pl.col("genres")).alias("genres_raw"),
            ]
        )
        .with_columns(
            [
                pl.col("genres_raw").str.split("|").alias("genres_list"),
                pl.when(pl.col("imdbId").is_not_null()).then(pl.format("tt{}", pl.col("imdbId"))).otherwise(None).alias("imdb_title_id"),
            ]
        )
        .sort("movieId")
    )

    duplicate_movie_ids = int(movies_clean.select(pl.col("movieId").is_duplicated().sum()).item())
    if duplicate_movie_ids != 0:
        raise ValueError(f"movies_clean has duplicate movieId values: {duplicate_movie_ids}")

    return movies_raw.height, movies_clean, movies_raw


def clean_ratings(raw_dir: Path) -> tuple[int, pl.DataFrame, pl.DataFrame]:
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

    duplicate_pairs = int(ratings_clean.select(pl.struct(["userId", "movieId"]).is_duplicated().sum()).item())
    if duplicate_pairs != 0:
        raise ValueError(f"ratings_clean has duplicate (userId, movieId) pairs: {duplicate_pairs}")

    return ratings_raw.height, ratings_clean, ratings_raw


def clean_tags(raw_dir: Path) -> tuple[int, pl.DataFrame, pl.DataFrame]:
    tags_raw = pl.read_csv(table_path(raw_dir, "tags"), infer_schema_length=10_000)
    tags_clean = (
        tags_raw.select(["userId", "movieId", "tag", "timestamp"])
        .with_columns(
            [
                to_int64_nullable("userId"),
                to_int64_nullable("movieId"),
                pl.col("tag").cast(pl.String, strict=False),
                to_int64_nullable("timestamp"),
            ]
        )
        .filter(
            pl.col("userId").is_not_null()
            & pl.col("movieId").is_not_null()
            & pl.col("tag").is_not_null()
            & pl.col("timestamp").is_not_null()
        )
        .with_columns(pl.col("tag").str.strip_chars().alias("tag"))
        .filter(pl.col("tag").str.len_chars() > 0)
        .with_columns(
            [
                pl.col("tag").str.to_lowercase().alias("tag_normalized"),
                pl.from_epoch("timestamp", time_unit="s").alias("tagged_at"),
            ]
        )
        .sort(["userId", "movieId", "timestamp"])
    )

    duplicate_rows = int(tags_clean.is_duplicated().sum())
    if duplicate_rows != 0:
        raise ValueError(f"tags_clean has duplicate full rows: {duplicate_rows}")

    return tags_raw.height, tags_clean, tags_raw


def build_movie_genres(movies_clean: pl.DataFrame) -> pl.DataFrame:
    return (
        movies_clean.select(["movieId", "genres_list"])
        .explode("genres_list")
        .rename({"genres_list": "genre"})
        .drop_nulls(["genre"])
        .sort(["movieId", "genre"])
    )


def write_outputs(
    processed_dir: Path,
    interim_dir: Path,
    movies_clean: pl.DataFrame,
    ratings_clean: pl.DataFrame,
    tags_clean: pl.DataFrame,
    movie_genres: pl.DataFrame,
    raw_inventory: pl.DataFrame,
    schema_profile: pl.DataFrame,
    missing_df: pl.DataFrame,
    quality_checks: dict[str, int],
    join_coverage_df: pl.DataFrame,
    scale_metrics: dict[str, int | float],
    table_memory_df: pl.DataFrame,
    row_changes: dict[str, dict[str, int]],
) -> None:
    movies_path = processed_dir / "movies_catalog.parquet"
    ratings_path = processed_dir / "ratings_clean.parquet"
    tags_path = processed_dir / "tags_clean.parquet"
    genres_path = processed_dir / "movie_genres.parquet"
    dictionary_path = processed_dir / "week03_processed_dictionary_profile.csv"
    report_path = processed_dir / "week03_cleaning_report.json"

    movies_clean.write_parquet(movies_path)
    ratings_clean.write_parquet(ratings_path)
    tags_clean.write_parquet(tags_path)
    movie_genres.write_parquet(genres_path)

    processed_dictionary = (
        schema_profile.join(missing_df, on=["table", "column"], how="left")
        .with_columns([
            pl.col("null_count").fill_null(0),
            pl.col("null_pct").fill_null(0.0),
        ])
        .sort(["table", "column"])
    )
    processed_dictionary.write_csv(dictionary_path)

    report = {
        "input_tables": raw_inventory.to_dicts(),
        "quality_checks": quality_checks,
        "join_checks": join_coverage_df.to_dicts(),
        "row_changes": row_changes,
        "key_decisions": [
            "Merged links into movies_catalog for direct imdbId/tmdbId access",
            "Preserved null tmdbId values and documented them as expected source incompleteness",
            "Deferred genome tables to Week 5 for representation work",
        ],
        "output_tables": [str(movies_path), str(ratings_path), str(tags_path), str(genres_path), str(dictionary_path)],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    (interim_dir / "week03_eda_summary.json").write_text(
        json.dumps(
            {
                "table_shapes": raw_inventory.to_dicts(),
                "missing_columns_nonzero": missing_df.filter(pl.col("null_count") > 0).to_dicts(),
                "quality_checks": quality_checks,
                "join_coverage": join_coverage_df.to_dicts(),
                "scale_metrics": scale_metrics,
                "table_memory_estimate_mb": table_memory_df.to_dicts(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (interim_dir / "week03_scale_metrics.json").write_text(
        json.dumps({"scale_metrics": scale_metrics, "table_memory_estimate_mb": table_memory_df.to_dicts()}, indent=2),
        encoding="utf-8",
    )
    processed_dictionary.write_csv(interim_dir / "week03_dictionary_profile.csv")

    print(f"Wrote processed outputs to {processed_dir}")
    print(f"Wrote interim profiling outputs to {interim_dir}")


def main() -> None:
    args = parse_args()
    ensure_directories(args.raw_dir, args.processed_dir, args.interim_dir)
    ensure_dataset(args.raw_dir, args.download_url, args.force_download, args.skip_download, args.keep_archive)

    raw_inventory = build_raw_inventory(args.raw_dir)
    schema_profile = build_schema_profile(args.raw_dir)
    missing_df = build_missing_profile(args.raw_dir)
    quality_checks = build_quality_checks(args.raw_dir)
    join_coverage_df = build_join_coverage(args.raw_dir)
    scale_metrics, table_memory_df = build_scale_metrics(args.raw_dir)

    movies_raw_rows, movies_clean, movies_raw = clean_movies(args.raw_dir)
    ratings_raw_rows, ratings_clean, ratings_raw = clean_ratings(args.raw_dir)
    tags_raw_rows, tags_clean, tags_raw = clean_tags(args.raw_dir)
    movie_genres = build_movie_genres(movies_clean)

    print(f"Loaded raw rows: movies={movies_raw_rows}, ratings={ratings_raw_rows}, tags={tags_raw_rows}")
    print(f"Cleaned rows: movies={movies_clean.height}, ratings={ratings_clean.height}, tags={tags_clean.height}")

    row_changes = {
        "movies": {
            "raw_rows": movies_raw.height,
            "clean_rows": movies_clean.height,
            "rows_removed": movies_raw.height - movies_clean.height,
        },
        "ratings": {
            "raw_rows": ratings_raw.height,
            "clean_rows": ratings_clean.height,
            "rows_removed": ratings_raw.height - ratings_clean.height,
        },
        "tags": {
            "raw_rows": tags_raw.height,
            "clean_rows": tags_clean.height,
            "rows_removed": tags_raw.height - tags_clean.height,
        },
    }

    write_outputs(
        args.processed_dir,
        args.interim_dir,
        movies_clean,
        ratings_clean,
        tags_clean,
        movie_genres,
        raw_inventory,
        schema_profile,
        missing_df,
        quality_checks,
        join_coverage_df,
        scale_metrics,
        table_memory_df,
        row_changes,
    )


if __name__ == "__main__":
    main()
