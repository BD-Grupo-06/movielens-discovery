#!/usr/bin/env python3
import argparse
from pathlib import Path
from src.pipelines.week05 import run_week05_pipeline
from src.config import DEFAULT_RANDOM_STATE

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 5 Pipeline Wrapper")
    parser.add_argument("--input-dir", default="data/interim", help="Input data dir")
    parser.add_argument("--processed-dir", default="data/processed", help="Output data dir")
    parser.add_argument("--artifacts-dir", default="artifacts/week05", help="Artifacts dir")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    run_week05_pipeline(
        data_dir=Path(args.input_dir),
        processed_dir=Path(args.processed_dir),
        artifacts_dir=Path(args.artifacts_dir),
        random_state=args.random_state
    )

if __name__ == "__main__":
    main()
