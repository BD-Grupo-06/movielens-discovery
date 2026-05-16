import json
import polars as pl
from pathlib import Path
from src.config import DATASET_DIR_NAME, RAW_FILES

def table_path(raw_dir: Path, table_name: str) -> Path:
    return raw_dir / DATASET_DIR_NAME / RAW_FILES[table_name]

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
        if not path.exists(): continue
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

    # Missing values profiling
    missing_rows = []
    for name, path in tables.items():
        if not path.exists(): continue
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

    # Simplified summary for export
    summary = {
        "table_shapes": shape_df.to_dicts(),
        "missing_columns_nonzero": missing_df.filter(pl.col("null_count") > 0).to_dicts(),
    }
    
    (interim_dir / "week03_eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    schema_df.write_csv(interim_dir / "week03_schema_profile.csv")
