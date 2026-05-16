import json
from pathlib import Path
import polars as pl
from src.config import MOVIE_LENS_URL, ARCHIVE_NAME, CORE_TABLES
from src.io import ensure_dir, download_file, safe_extract
from src.cleaning import clean_movies, clean_ratings, clean_tags, build_movie_genres
from src.profiling import build_eda_artifacts, profile_columns

def run_week03_pipeline(
    raw_dir: Path,
    processed_dir: Path,
    interim_dir: Path,
    force_download: bool = False,
    skip_download: bool = False,
    keep_archive: bool = False
):
    ensure_dir(raw_dir)
    ensure_dir(processed_dir)
    ensure_dir(interim_dir)

    archive_path = raw_dir / ARCHIVE_NAME
    
    # Download and extract if necessary
    if not skip_download:
        if force_download or not archive_path.exists():
            print(f"Downloading dataset from {MOVIE_LENS_URL}")
            download_file(MOVIE_LENS_URL, archive_path)
            print("Extracting dataset...")
            safe_extract(archive_path, raw_dir)
            if not keep_archive:
                archive_path.unlink()

    # Profiling
    print("Building EDA artifacts...")
    build_eda_artifacts(raw_dir, interim_dir)

    # Cleaning
    print("Cleaning data...")
    movies_raw, links_raw, movies_clean = clean_movies(raw_dir)
    ratings_raw, ratings_clean = clean_ratings(raw_dir)
    tags_raw, tags_clean = clean_tags(raw_dir)

    # Join checks
    join_checks = {
        "ratings_unmatched": int(
            ratings_clean.select("movieId")
            .unique()
            .join(movies_clean.select("movieId").unique(), on="movieId", how="anti")
            .height
        ),
        "tags_unmatched": int(
            tags_clean.select("movieId")
            .unique()
            .join(movies_clean.select("movieId").unique(), on="movieId", how="anti")
            .height
        ),
    }

    # Save processed data
    print(f"Saving intermediate data to {interim_dir}...")
    movies_clean.write_parquet(interim_dir / "movies_catalog.parquet")
    ratings_clean.write_parquet(interim_dir / "ratings_clean.parquet")
    tags_clean.write_parquet(interim_dir / "tags_clean.parquet")

    # Final report
    report = {
        "row_changes": [
            {"table": "movies", "raw": movies_raw.height, "clean": movies_clean.height},
            {"table": "ratings", "raw": ratings_raw.height, "clean": ratings_clean.height},
            {"table": "tags", "raw": tags_raw.height, "clean": tags_clean.height},
        ],
        "join_checks": join_checks,
        "key_decisions": [
            "Merged links into movies_catalog for direct imdbId/tmdbId access",
            "Preserved null tmdbId values where missing in source",
        ]
    }
    (interim_dir / "week03_cleaning_report.json").write_text(json.dumps(report, indent=2))
    print("Week 03 pipeline completed successfully.")

