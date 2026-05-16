#!/usr/bin/env python3
import argparse
from pathlib import Path
from src.pipelines.week03 import run_week03_pipeline
from src.config import MOVIE_LENS_URL

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 3 Pipeline Wrapper")
    parser.add_argument("--download-url", default=MOVIE_LENS_URL)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    run_week03_pipeline(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        interim_dir=args.interim_dir,
        force_download=args.force_download,
        skip_download=args.skip_download,
        keep_archive=args.keep_archive
    )

if __name__ == "__main__":
    main()
