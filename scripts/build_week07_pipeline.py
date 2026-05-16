import argparse
from pathlib import Path
from src.pipelines.week07 import run_week07_pipeline
from src.config import DEFAULT_RANDOM_STATE

def parse_args():
    parser = argparse.ArgumentParser(description="Week 7 Pipeline Wrapper")
    parser.add_argument("--input-dir", default="data/processed", help="Input data dir")
    parser.add_argument("--processed-dir", default="data/processed", help="Output data dir")
    parser.add_argument("--artifacts-dir", default="artifacts/week07", help="Artifacts dir")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)

    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Run only report/insights generation step"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    if args.report_only:
        from src.pipelines.week07 import generate_report_insights

        generate_report_insights(
            processed_dir=Path(args.processed_dir),
            data_dir=Path("data/interim")  # o el que uses
        )
        return

    run_week07_pipeline(
        processed_dir=Path(args.processed_dir),
        artifacts_dir=Path(args.artifacts_dir),
        random_state=args.random_state
    )

if __name__ == "__main__":
    main()