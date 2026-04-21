# movielens-discovery

Semester project for Big Data: Domain Discovery, Recommendation, and Graph Intelligence using the MovieLens 25M dataset.

## Objective

This project aims to build an end-to-end movie discovery system, not just a single model. We start from raw MovieLens 25M files, produce versioned cleaned datasets, and progressively build the semester layers: representation, clustering, recommendation, and graph analytics.

What we are planning to build across the course:

1. A reliable data pipeline from ingestion to processed artifacts.
2. Feature representations that combine ratings, tags, genres, and temporal behavior.
3. Movie segmentation with validated clustering experiments.
4. A recommendation engine with baseline and stronger ranking models.
5. A graph-based analysis layer (centrality, structure, and item relationships).

Core architecture layers:

1. Catalog layer: movies metadata.
2. Feature layer: content and behavioral features.
3. Interaction layer: user ratings/tags and co-occurrence.
4. Graph layer: item-item and/or user-item projections.
5. Pipeline layer: reproducible path from raw data to processed artifacts.

## Quick Start

### 1. Install requirements

Download and install project requirements:

```bash
python3 -m pip install -r requirements.txt
```

### 2. Build Week 3 outputs (dataset already present)

If `data/raw/ml-25m/` already exists, run:

```bash
python3 scripts/build_week03_pipeline.py --skip-download
```

### 3. Build Week 3 outputs (download + extraction + cleaning)

If raw files are not present yet, run:

```bash
python3 scripts/build_week03_pipeline.py
```

This command handles download, extraction, cleaning, and profiling in one run.

### 4. Optional legacy download-only script

Run this command before anything else:

```bash
./scripts/download_dataset.sh
```

This script populates `data/raw/ml-25m/` with the MovieLens files used by the project.

Main outputs written by the pipeline:

- `data/processed/week03_v1/movies_catalog.parquet`
- `data/processed/week03_v1/ratings_clean.parquet`
- `data/processed/week03_v1/tags_clean.parquet`
- `data/processed/week03_v1/movie_genres.parquet`
- `data/processed/week03_v1/week03_cleaning_report.json`
- `data/processed/week03_v1/week03_processed_dictionary_profile.csv`
- `data/interim/week03_eda_summary.json`
- `data/interim/week03_scale_metrics.json`
- `data/interim/week03_dictionary_profile.csv`

## Project Structure

```text
movielens-discovery/
	data/
		raw/
		interim/
		processed/
	notebooks/
	src/
	scripts/
	reports/
	artifacts/
	README.md
```

## Reproducibility Notes

- Keep raw data immutable under `data/raw/`.
- Write cleaned/transformed tables to `data/interim/` and `data/processed/`.
- Prefer scripts in `scripts/` and code in `src/` over notebook-only workflows.
- Save experiment outputs, plots, and model artifacts in `artifacts/`.

## Assignment Deliverables Mapping

This repository is organized to support the milestone flow:

- Week 3: dataset charter, schema draft, processed dataset v1, data dictionary draft.
- Week 5: feature representation and dimensionality reduction report.
- Week 7: clustering experiments and validation.
- Week 10: recommendation/ranking baseline and advanced approach with offline evaluation.
- Week 12: graph construction and centrality analysis.
- Week 14: integrated final report, reproducible runbook, and demo artifact.

## Ethics and Access Note

- Data source: GroupLens MovieLens 25M public release.
- Usage: educational and research work for this course assignment.
- Personal-data risk: user identifiers are anonymized IDs in the source dataset.
- Mitigation: do not attempt re-identification and do not add external personal identifiers.
