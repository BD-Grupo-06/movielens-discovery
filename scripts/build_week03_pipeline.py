#!/usr/bin/env python3
"""End-to-end Week 3 pipeline aligned with EDA + cleaning notebooks."""

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
    parser = argparse.ArgumentParser(description="Download, profile, clean, and export Week 3 MovieLens artifacts.")
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


def ensure_all_tables_present(raw_dir: Path) -> None:
    missing = [str(table_path(raw_dir, table_name)) for table_name in ALL_TABLES if not table_path(raw_dir, table_name).exists()]
    if missing:
        raise FileNotFoundError("Missing expected MovieLens tables: " + ", ".join(missing))


def to_int64_nullable(column_name: str) -> pl.Expr:
    return pl.col(column_name).cast(pl.Int64, strict=False)


def to_float64(column_name: str) -> pl.Expr:
    return pl.col(column_name).cast(pl.Float64, strict=False)


def profile_columns(df: pl.DataFrame, table_name: str) -> pl.DataFrame:
    rows = df.height
    return pl.DataFrame(
        [
            {
                "table": table_name,
                "column": col,
                "dtype": str(dtype),
                "null_count": int(df.select(pl.col(col).is_null().sum()).item()),
                "null_pct": round((df.select(pl.col(col).is_null().sum()).item() / rows) * 100, 4) if rows else 0.0,
            }
            for col, dtype in df.schema.items()
        ]
    )


def build_raw_profile(raw_dir: Path) -> pl.DataFrame:
    rows = []
    for table_name in ["ratings", "movies", "tags", "links"]:
        path = table_path(raw_dir, table_name)
        rows.append(
            {
                "table": table_name,
                "rows": pl.scan_csv(path).select(pl.len()).collect().item(),
                "size_mb": round(path.stat().st_size / (1024**2), 2),
            }
        )
    return pl.DataFrame(rows)


def build_eda_artifacts(raw_dir: Path, interim_dir: Path) -> None:
    core_tables = {
        "movies": table_path(raw_dir, "movies"),
        "ratings": table_path(raw_dir, "ratings"),
        "tags": table_path(raw_dir, "tags"),
        "links": table_path(raw_dir, "links"),
    }
    optional_tables = {
        "genome_scores": table_path(raw_dir, "genome_scores"),
        "genome_tags": table_path(raw_dir, "genome_tags"),
    }
    tables = {**core_tables, **optional_tables}

    shape_rows: list[dict[str, object]] = []
    schema_rows: list[dict[str, object]] = []

    for name, path in tables.items():
        lf = pl.scan_csv(path, infer_schema_length=10_000)
        n_rows = lf.select(pl.len().alias("rows")).collect().item()
        sample = pl.read_csv(path, n_rows=2_000, infer_schema_length=10_000)

        shape_rows.append(
            {
                "table": name,
                "rows": n_rows,
                "cols": sample.width,
                "size_mb": round(path.stat().st_size / (1024**2), 2),
                "scope": "core" if name in core_tables else "optional",
            }
        )

        for column_name, dtype in sample.schema.items():
            schema_rows.append(
                {
                    "table": name,
                    "column": column_name,
                    "sample_dtype": str(dtype),
                }
            )

    shape_df = pl.DataFrame(shape_rows).sort("rows", descending=True)
    schema_df = pl.DataFrame(schema_rows)

    table_usage_recommendation = pl.DataFrame(
        [
            {"table": "ratings", "week3_use": "required", "reason": "primary interaction layer and sparsity analysis"},
            {"table": "movies", "week3_use": "required", "reason": "catalog layer and metadata base"},
            {"table": "tags", "week3_use": "recommended", "reason": "text signal for future content and hybrid features"},
            {"table": "links", "week3_use": "recommended", "reason": "external IDs and integration hooks"},
            {"table": "genome_scores", "week3_use": "optional", "reason": "large feature source for week5+, can be deferred"},
            {"table": "genome_tags", "week3_use": "optional", "reason": "lookup table for genome_scores tags"},
        ]
    )

    missing_rows = []
    for name, path in tables.items():
        lf = pl.scan_csv(path, infer_schema_length=10_000)
        n_rows = lf.select(pl.len()).collect().item()
        columns = pl.read_csv(path, n_rows=5).columns

        null_counts = lf.select([pl.col(column).is_null().sum().alias(column) for column in columns]).collect()
        for column in columns:
            n_null = int(null_counts[0, column])
            missing_rows.append(
                {
                    "table": name,
                    "column": column,
                    "null_count": n_null,
                    "null_pct": round((n_null / n_rows) * 100, 4) if n_rows else 0.0,
                }
            )

    missing_df = pl.DataFrame(missing_rows)

    movies = pl.read_csv(table_path(raw_dir, "movies"), infer_schema_length=10_000)
    ratings = pl.read_csv(table_path(raw_dir, "ratings"), infer_schema_length=10_000)
    tags = pl.read_csv(table_path(raw_dir, "tags"), infer_schema_length=10_000)
    links = pl.read_csv(table_path(raw_dir, "links"), infer_schema_length=10_000)
    genome_scores = pl.read_csv(table_path(raw_dir, "genome_scores"), infer_schema_length=10_000)
    genome_tags = pl.read_csv(table_path(raw_dir, "genome_tags"), infer_schema_length=10_000)

    checks = {
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

    movies_ids = movies.select("movieId").unique()
    ratings_ids = ratings.select("movieId").unique()
    tags_ids = tags.select("movieId").unique()
    genome_movie_ids = genome_scores.select("movieId").unique()
    genome_tag_ids = genome_scores.select("tagId").unique()

    orphan_ratings_movie_ids = ratings_ids.join(movies_ids, on="movieId", how="anti").height
    orphan_tags_movie_ids = tags_ids.join(movies_ids, on="movieId", how="anti").height
    orphan_genome_movie_ids = genome_movie_ids.join(movies_ids, on="movieId", how="anti").height
    orphan_genome_tag_ids = genome_tag_ids.join(genome_tags.select("tagId").unique(), on="tagId", how="anti").height

    join_coverage_df = pl.DataFrame(
        {
            "relationship_check": [
                "ratings movieId matched in movies",
                "tags movieId matched in movies",
                "genome_scores movieId matched in movies",
                "genome_scores tagId matched in genome_tags",
            ],
            "unmatched_count": [
                int(orphan_ratings_movie_ids),
                int(orphan_tags_movie_ids),
                int(orphan_genome_movie_ids),
                int(orphan_genome_tag_ids),
            ],
            "coverage_pct": [
                round((1 - (orphan_ratings_movie_ids / max(1, ratings_ids.height))) * 100, 4),
                round((1 - (orphan_tags_movie_ids / max(1, tags_ids.height))) * 100, 4),
                round((1 - (orphan_genome_movie_ids / max(1, genome_movie_ids.height))) * 100, 4),
                round((1 - (orphan_genome_tag_ids / max(1, genome_tag_ids.height))) * 100, 4),
            ],
        }
    )

    users_n = ratings.select(pl.col("userId").n_unique()).item()
    items_n = ratings.select(pl.col("movieId").n_unique()).item()
    interactions_n = ratings.height
    possible_interactions = int(users_n * items_n)
    sparsity = 1 - (interactions_n / possible_interactions)

    scale_metrics = {
        "users": int(users_n),
        "items": int(items_n),
        "interactions": int(interactions_n),
        "possible_user_item_pairs": possible_interactions,
        "density_pct": round((interactions_n / possible_interactions) * 100, 6),
        "sparsity_pct": round(sparsity * 100, 6),
    }

    table_memory_df = shape_df.select(["table", "rows", "cols", "size_mb", "scope"]).sort("size_mb", descending=True)

    rating_dist = ratings.group_by("rating").len().sort("rating")
    top_genres = (
        movies.with_columns(pl.col("genres").str.split("|"))
        .explode("genres")
        .group_by("genres")
        .len()
        .sort("len", descending=True)
        .head(15)
    )

    summary = {
        "table_shapes": shape_df.to_dicts(),
        "recommended_table_usage": table_usage_recommendation.to_dicts(),
        "missing_columns_nonzero": missing_df.filter(pl.col("null_count") > 0).sort(["table", "null_count"], descending=True).to_dicts(),
        "quality_checks": checks,
        "join_coverage": join_coverage_df.to_dicts(),
        "scale_metrics": scale_metrics,
        "table_memory_estimate_mb": table_memory_df.to_dicts(),
        "rating_distribution": rating_dist.to_dicts(),
        "top_genres": top_genres.to_dicts(),
    }

    summary_path = interim_dir / "week03_eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    scale_path = interim_dir / "week03_scale_metrics.json"
    scale_path.write_text(
        json.dumps({"scale_metrics": scale_metrics, "table_memory_estimate_mb": table_memory_df.to_dicts()}, indent=2),
        encoding="utf-8",
    )

    dictionary_profile_df = (
        schema_df.join(
            missing_df.select(["table", "column", "null_count", "null_pct"]),
            on=["table", "column"],
            how="left",
        )
        .with_columns(
            [
                pl.col("null_count").fill_null(0),
                pl.col("null_pct").fill_null(0.0),
            ]
        )
        .sort(["table", "column"])
    )
    dictionary_profile_df.write_csv(interim_dir / "week03_dictionary_profile.csv")


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

    duplicate_pairs = int(ratings_clean.select(pl.struct(["userId", "movieId"]).is_duplicated().sum()).item())
    if duplicate_pairs != 0:
        raise ValueError(f"ratings_clean has duplicate (userId, movieId) pairs: {duplicate_pairs}")

    return ratings_raw, ratings_clean


def clean_tags(raw_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
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

    return tags_raw, tags_clean


def build_movie_genres(movies_clean: pl.DataFrame) -> pl.DataFrame:
    return (
        movies_clean.select(["movieId", "genres_list"])
        .explode("genres_list")
        .rename({"genres_list": "genre"})
        .drop_nulls(["genre"])
        .sort(["movieId", "genre"])
    )


def to_notebook_style_path(path: Path) -> str:
    as_posix = path.as_posix()
    if path.is_absolute() or as_posix.startswith("../"):
        return as_posix
    return f"../{as_posix}"


def write_cleaning_artifacts(processed_dir: Path, raw_dir: Path) -> None:
    raw_profile = build_raw_profile(raw_dir)

    movies_raw, links_raw, movies_clean = clean_movies(raw_dir)
    ratings_raw, ratings_clean = clean_ratings(raw_dir)
    tags_raw, tags_clean = clean_tags(raw_dir)
    movie_genres = build_movie_genres(movies_clean)

    join_checks = pl.DataFrame(
        {
            "relationship": [
                "ratings.movieId in movies_clean.movieId",
                "tags.movieId in movies_clean.movieId",
            ],
            "unmatched_count": [
                int(
                    ratings_clean.select("movieId")
                    .unique()
                    .join(movies_clean.select("movieId").unique(), on="movieId", how="anti")
                    .height
                ),
                int(
                    tags_clean.select("movieId")
                    .unique()
                    .join(movies_clean.select("movieId").unique(), on="movieId", how="anti")
                    .height
                ),
            ],
        }
    )

    before_after = pl.DataFrame(
        [
            {
                "table": "movies",
                "raw_rows": movies_raw.height,
                "clean_rows": movies_clean.height,
                "rows_removed": movies_raw.height - movies_clean.height,
            },
            {
                "table": "ratings",
                "raw_rows": ratings_raw.height,
                "clean_rows": ratings_clean.height,
                "rows_removed": ratings_raw.height - ratings_clean.height,
            },
            {
                "table": "tags",
                "raw_rows": tags_raw.height,
                "clean_rows": tags_clean.height,
                "rows_removed": tags_raw.height - tags_clean.height,
            },
            {
                "table": "links_as_joined_columns",
                "raw_rows": links_raw.height,
                "clean_rows": movies_clean.select(["movieId", "imdbId", "tmdbId"]).height,
                "rows_removed": links_raw.height - movies_clean.select(["movieId", "imdbId", "tmdbId"]).height,
            },
        ]
    )

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

    processed_dictionary = pl.concat(
        [
            profile_columns(movies_clean, "movies_catalog"),
            profile_columns(ratings_clean, "ratings_clean"),
            profile_columns(tags_clean, "tags_clean"),
            profile_columns(movie_genres, "movie_genres"),
        ],
        how="vertical",
    )
    processed_dictionary.sort(["table", "column"]).write_csv(dictionary_path)

    cleaning_report = {
        "input_tables": raw_profile.to_dicts(),
        "row_changes": before_after.to_dicts(),
        "join_checks": join_checks.to_dicts(),
        "key_decisions": [
            "Merged links into movies_catalog for direct imdbId/tmdbId access",
            "Preserved null tmdbId values and documented them as expected source incompleteness",
            "Deferred genome tables to Week 5 for representation work",
        ],
        "output_tables": [
            to_notebook_style_path(movies_path),
            to_notebook_style_path(ratings_path),
            to_notebook_style_path(tags_path),
            to_notebook_style_path(genres_path),
            to_notebook_style_path(dictionary_path),
        ],
    }
    report_path.write_text(json.dumps(cleaning_report, indent=2), encoding="utf-8")

    print(f"Loaded raw rows: movies={movies_raw.height}, ratings={ratings_raw.height}, tags={tags_raw.height}")
    print(f"Cleaned rows: movies={movies_clean.height}, ratings={ratings_clean.height}, tags={tags_clean.height}")


def main() -> None:
    args = parse_args()
    ensure_directories(args.raw_dir, args.processed_dir, args.interim_dir)
    ensure_dataset(args.raw_dir, args.download_url, args.force_download, args.skip_download, args.keep_archive)
    ensure_all_tables_present(args.raw_dir)

    build_eda_artifacts(args.raw_dir, args.interim_dir)
    write_cleaning_artifacts(args.processed_dir, args.raw_dir)

    print(f"Wrote processed outputs to {args.processed_dir}")
    print(f"Wrote interim profiling outputs to {args.interim_dir}")


if __name__ == "__main__":
    main()
