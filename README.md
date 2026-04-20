# movielens-discovery

Semester project for Big Data: Domain Discovery, Recommendation, and Graph Intelligence using the MovieLens 25M dataset.

## Objective

Build a reproducible discovery and recommendation pipeline with the required course layers:

1. Catalog layer: movies metadata.
2. Feature layer: content and behavioral features.
3. Interaction layer: user ratings/tags and co-occurrence.
4. Graph layer: item-item and/or user-item projections.
5. Pipeline layer: reproducible path from raw data to processed artifacts.

## Quick Start

### 1) Download the dataset (first step)

Run this command before anything else:

```bash
./scripts/download_dataset.sh
```

This script populates `data/raw/ml-25m/` with the MovieLens files used by the project.

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
