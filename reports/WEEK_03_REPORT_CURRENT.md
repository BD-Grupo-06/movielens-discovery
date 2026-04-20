# Week 3 Milestone Report: Dataset Charter and Processed Dataset V1

## Executive Summary

This comprehensive report consolidates all seven Week 3 deliverables for the semester group assignment:

- **Project title**: Personalized Movie Discovery and Recommendation Engine (MovieLens 25M)
- **Course**: Semester Project: Domain Discovery, Recommendation, and Graph Intelligence
- **Milestone objective**: Prove valid domain, valid data source, and reproducible first dataset build
- **Report date**: April 2026
- **Status**: 6 of 7 deliverables completed; processed dataset V1 pending final pipeline execution

---

## 1. Project Proposal

### Project Title

Personalized Movie Discovery and Recommendation Engine (MovieLens 25M)

### Domain

Entertainment recommendation systems.

### Problem Statement

Users face a large movie catalog and need relevant, personalized discovery support. The project builds a data product that starts from raw interaction data and produces ranked movie recommendations, interpretable movie segments, and graph-based insights.

### Expected Product Questions

1. Which movies are similar in feature and behavior space?
2. Which movies belong to the same latent segment?
3. Which movies should be recommended next for each user?
4. Which movies are central in the interaction graph?

### Why MovieLens 25M Is Suitable for the Full Course

1. **Non-trivial scale**: 25M ratings, 162K users, 59K movies
2. **Multi-table structure**: catalog, interactions, tags, genome relevance, external links
3. **Strong fit to course layers**:
    - Catalog layer (movies)
    - Feature layer (tags, genome, derived aggregates)
    - Interaction layer (ratings)
    - Graph layer (user-item projections, item-item similarity)
    - Pipeline layer (reproducible ingestion and transformation)
4. **Clear provenance and academic use terms** from GroupLens Research
5. **Enables full semester arc**: from ingestion → representation → clustering → recommendation → graph analytics → defense

### Week 3 Scope Note

This milestone only delivers the initial reproducible data build and documentation (processed dataset V1). Model training and advanced analytics are out of scope for Week 3 and will proceed in Weeks 5–14.

---

## 2. Source Inventory

### Primary Dataset

- **Dataset name**: MovieLens 25M
- **Official page**: https://grouplens.org/datasets/movielens/25m/
- **Direct download**: https://files.grouplens.org/datasets/movielens/ml-25m.zip
- **Dataset README**: https://files.grouplens.org/datasets/movielens/ml-25m-README.html

### License and Access Conditions

- **Provider**: GroupLens Research, University of Minnesota
- **Use condition summary**: Research and educational use; attribution required; commercial use requires permission
- **Access method**: Direct public download (no account required)

### Citation

Harper, F. M., and Konstan, J. A. (2015). The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS), 5(4), 19:1-19:19.

### Raw File Formats

The ZIP archive includes CSV files:

1. `ratings.csv` — user-movie interactions
2. `movies.csv` — catalog metadata
3. `tags.csv` — user-generated tags
4. `links.csv` — external identifiers (IMDB, TMDB)
5. `genome-scores.csv` — tag relevance matrix
6. `genome-tags.csv` — tag dictionary

### Estimated Storage

- **Compressed**: ~250 MB
- **Uncompressed working size**: ~1.0–1.5 GB (including interim outputs)

### Week 3 Compliance

This project uses a single official source, so no cross-source matching logic is required for Week 3.

---

## 3. Schema Draft

### Entity Tables

#### movies

- `movieId` (int, primary key): numeric identifier
- `title` (string): movie title, often includes release year in parentheses
- `genres` (string): pipe-separated genre categories

#### ratings

- `userId` (int): anonymized user identifier
- `movieId` (int, foreign key → movies.movieId): movie identifier
- `rating` (float): user's numeric rating [0.5, 5.0], half-star granularity
- `timestamp` (int): Unix epoch seconds when rating was created
- Candidate primary key: `(userId, movieId)` — one rating per user-movie pair

#### tags

- `userId` (int): anonymized user identifier
- `movieId` (int, foreign key → movies.movieId): movie identifier
- `tag` (string): user-generated tag (e.g., "funny", "action")
- `timestamp` (int): Unix epoch seconds when tag was created

#### genome-tags

- `tagId` (int, primary key): numeric tag identifier
- `tag` (string): standardized tag string from tag genome

#### genome-scores

- `movieId` (int, foreign key → movies.movieId): movie identifier
- `tagId` (int, foreign key → genome-tags.tagId): tag identifier
- `relevance` (float): tag relevance score [0.0, 1.0]

#### links

- `movieId` (int, primary key, foreign key → movies.movieId): movie identifier
- `imdbId` (int): IMDB identifier
- `tmdbId` (int): TMDB (The Movie Database) identifier

### Expected Joins

1. `ratings LEFT JOIN movies` ON `ratings.movieId = movies.movieId`
2. `tags LEFT JOIN movies` ON `tags.movieId = movies.movieId`
3. `genome-scores LEFT JOIN movies` ON `genome-scores.movieId = movies.movieId`
4. `genome-scores LEFT JOIN genome-tags` ON `genome-scores.tagId = genome-tags.tagId`
5. `movies LEFT JOIN links` ON `movies.movieId = links.movieId`

### Processed Week 3 Outputs

1. **ratings_cleaned.parquet**
    - Based on ratings.csv after range/null/duplicate checks
    - Columns: `userId`, `movieId`, `rating`, `timestamp`

2. **movies_with_features.parquet**
    - Movies joined with aggregated rating statistics
    - Base columns: `movieId`, `title`, `genres`
    - Derived columns: `release_year`, `genre_count`, `primary_genre`, `rating_count`, `rating_mean`, `rating_std`, `rating_min`, `rating_max`, `first_rating_ts`, `last_rating_ts`

### Key Data Quality Checks for Week 3

1. Rating values are within [0.5, 5.0]
2. Duplicate `(userId, movieId)` records handled consistently
3. Required IDs (`userId`, `movieId`) are non-null
4. Join coverage between ratings.movieId and movies.movieId is reported as 100%

---

## 4. Processed Dataset V1 Status

### Current Status: Pending Final Pipeline Execution

#### What Exists

- Reproducible ingestion command documented in project `README.md`
- EDA and data profiling completed in `notebooks/week03_eda.ipynb`
- Interim quality-check artifacts generated in `data/interim/`

#### What Is Needed

Run the pipeline to generate processed outputs:

```bash
python src/pipeline.py \
  --raw-dir data/raw/ml-25m \
  --interim-dir data/interim \
  --processed-dir data/processed
```

#### Expected Outputs

After successful pipeline execution:

- `data/processed/ratings_cleaned.parquet` — cleaned interaction table
- `data/processed/movies_with_features.parquet` — enriched catalog table
- `data/processed/data_quality_report.json` — detailed quality metrics (optional)

---

## 5. Data Dictionary Draft

This data dictionary defines all columns in processed Week 3 outputs. Final null rates and type confirmations will be populated after pipeline execution.

### ratings_cleaned.parquet

| Column    | Type    | Description                    | Allowed/Observed Range | Null Rate   | Notes                    |
| --------- | ------- | ------------------------------ | ---------------------- | ----------- | ------------------------ |
| userId    | int64   | Anonymized user identifier     | ≥ 1                    | 0% expected | Source: ratings.csv      |
| movieId   | int64   | Movie identifier               | ≥ 1                    | 0% expected | Joins to movies.movieId  |
| rating    | float64 | Explicit user rating           | [0.5, 5.0]             | 0% expected | Half-star granularity    |
| timestamp | int64   | Rating event time (Unix epoch) | dataset-dependent      | 0% expected | Seconds since 1970-01-01 |

### movies_with_features.parquet

| Column          | Type    | Description                 | Allowed/Observed Range | Null Rate         | Notes                                      |
| --------------- | ------- | --------------------------- | ---------------------- | ----------------- | ------------------------------------------ |
| movieId         | int64   | Movie identifier            | ≥ 1                    | 0% expected       | Primary catalog key                        |
| title           | string  | Movie title                 | text                   | 0% expected       | Often includes release year in parentheses |
| genres          | string  | Pipe-separated genres       | text                   | 0% expected       | Example: "Action\|Adventure"               |
| release_year    | float64 | Parsed release year         | 1900–2025              | to fill after run | NaN when pattern missing                   |
| genre_count     | int64   | Number of listed genres     | ≥ 1                    | 0% expected       | Derived feature                            |
| primary_genre   | string  | First genre token           | text                   | to fill after run | Derived feature                            |
| rating_count    | int64   | Number of ratings for movie | ≥ 0                    | 0% expected       | Aggregated from ratings_cleaned            |
| rating_mean     | float64 | Mean movie rating           | [0.5, 5.0]             | to fill after run | NaN if unrated                             |
| rating_std      | float64 | Rating standard deviation   | ≥ 0                    | to fill after run | NaN if single-rating movie                 |
| rating_min      | float64 | Minimum movie rating        | [0.5, 5.0]             | to fill after run | Derived from ratings                       |
| rating_max      | float64 | Maximum movie rating        | [0.5, 5.0]             | to fill after run | Derived from ratings                       |
| first_rating_ts | float64 | Earliest rating timestamp   | dataset-dependent      | to fill after run | NaN if unrated                             |
| last_rating_ts  | float64 | Latest rating timestamp     | dataset-dependent      | to fill after run | NaN if unrated                             |

### Validation Checklist

1. Confirm schema using `df.dtypes` on both processed outputs
2. Confirm null rates using `df.isna().mean() * 100`
3. Confirm rating bounds and duplicate rules in `ratings_cleaned.parquet`
4. Verify no `(userId, movieId)` duplicates in `ratings_cleaned.parquet`

---

## 6. Scale Analysis

### Dataset Scale (from EDA Profiling)

| Table         |       Rows | Columns | Size (MB) | Scope    |
| ------------- | ---------: | ------: | --------: | -------- |
| ratings       | 25,000,095 |       4 |    646.84 | core     |
| movies        |     62,423 |       3 |      2.90 | core     |
| tags          |  1,093,360 |       4 |     37.01 | core     |
| links         |     62,423 |       3 |      1.31 | core     |
| genome_scores | 15,584,448 |       3 |    415.00 | optional |
| genome_tags   |      1,128 |       2 |      0.02 | optional |

**Total uncompressed raw size**: ~1.1 GB

### Interaction Matrix Scale

- **Unique users**: 162,541
- **Unique items (movies with ratings)**: 59,047
- **Observed interactions**: 25,000,095
- **Possible user-item pairs**: 162,541 × 59,047 = 9,597,558,427
- **Density**: 25,000,095 / 9,597,558,427 = **0.2605%**
- **Sparsity**: 1 − 0.002605 = **99.7395%**

This high sparsity is typical for recommendation domains and justifies collaborative filtering and matrix-factorization approaches later in the course.

### Data Quality Checks Executed

| Check                                                | Result           | Notes                        |
| ---------------------------------------------------- | ---------------- | ---------------------------- |
| Required fields (userId, movieId) missing in ratings | 0                | 100% coverage                |
| Ratings outside [0.5, 5.0]                           | 0                | All ratings valid            |
| Duplicate (userId, movieId) pairs                    | 0                | No duplicate interactions    |
| Missing tmdbId in links                              | 107 rows (0.17%) | Acceptable; imdbId available |
| Join coverage: ratings → movies                      | 100%             | All rated movies in catalog  |
| Join coverage: tags → movies                         | 100%             | All tagged movies in catalog |
| Join coverage: genome-scores → movies                | 100%             | All genome movies in catalog |

### Evidence Files Generated

- `data/interim/week03_eda_summary.json` — Full profiling metrics
- `data/interim/week03_scale_metrics.json` — Scale and sparsity metrics
- `data/interim/week03_dictionary_profile.csv` — Column-level null/type analysis

### Table Scope Decision

**Week 3 (core/required):**

- ratings
- movies

**Week 3+ (recommended for Week 5+):**

- tags
- links

**Defer to later weeks:**

- genome_scores
- genome_tags

**Rationale**: Week 3 requires reproducible ingestion and cleaned base tables. Genome processing (high-dimensional tag relevance matrix) is deferred to dimensionality reduction phase (Week 5) without breaking reproducibility.

---

## 7. Ethics and Access Note

### 1) Data Origin

The project uses **MovieLens 25M**, provided by GroupLens Research at the University of Minnesota. The dataset consists of:

- Anonymized user interactions with movies (ratings and tags)
- Catalog metadata (titles, genres)
- Tag metadata and tag relevance scores
- External links (IMDB, TMDB identifiers)

### 2) Why the Team Is Allowed to Use It

- **Source is publicly distributed** for research and educational use
- **Team usage is academic and non-commercial**, consistent with terms
- **Final report will include proper attribution and citation**

Reference links:

- https://grouplens.org/datasets/movielens/25m/
- Harper, F. M., and Konstan, J. A. (2015). The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS), 5(4), 19:1-19:19.

### 3) Personal-Data Risks

- **No direct PII present**: Dataset contains no names, email addresses, phone numbers, or identifiable external references
- **Interaction histories as signals**: Timestamped ratings and tags reflect user behavior and preferences
- **Behavioral re-identification risk**: Detailed interaction patterns could theoretically increase re-identification risk if combined with external data

### 4) How Risks Are Reduced

1. **No attempt to identify or profile real individuals**: Analysis focuses on aggregate patterns and model behavior
2. **No external linkage for deanonymization**: Project does not cross-reference user IDs with other data sources
3. **No unauthorized redistribution**: Raw personal-level records remain within project team only
4. **Aggregate-only reporting**: Final outputs report segment and recommendation metrics, not individual user profiles
5. **Access control**: Working files restricted to authorized project team members
6. **Transparency through reproducibility**: All transformation steps documented and auditable

### Compliance Statement

This Week 3 milestone follows MovieLens source access conditions and includes an explicit reproducible ingestion pipeline for transparency and auditability.

---

## Final Week 3 Status

| Deliverable               | Status               | Evidence                       |
| ------------------------- | -------------------- | ------------------------------ |
| 1. Project proposal       | ✅ Complete          | Section 1 of this report       |
| 2. Source inventory       | ✅ Complete          | Section 2 of this report       |
| 3. Schema draft           | ✅ Complete          | Section 3 of this report       |
| 4. Processed dataset V1   | ⏳ Pending execution | See Section 4 pipeline command |
| 5. Data dictionary draft  | ✅ Complete          | Section 5 of this report       |
| 6. Scale analysis         | ✅ Complete          | Section 6 of this report       |
| 7. Ethics and access note | ✅ Complete          | Section 7 of this report       |

**Overall status**: 6 of 7 deliverables complete. The single remaining task is executing the pipeline to generate processed parquet files in `data/processed/`.

---

## Next Steps

### Immediate (to close Week 3)

1. Execute pipeline: `python src/pipeline.py --raw-dir data/raw/ml-25m --interim-dir data/interim --processed-dir data/processed`
2. Verify `ratings_cleaned.parquet` and `movies_with_features.parquet` exist
3. Fill any remaining "to fill after run" fields in the Data Dictionary (Section 5)
4. Confirm final file inventory matches submission checklist

### Week 4 Preparation

- Design feature engineering pipeline for Week 5 representation phase
- Identify key feature categories: temporal, rating-aggregate, tag-based, genre-based

### Week 5 and Beyond

- Dimensionality reduction (PCA, SVD, optional t-SNE)
- Clustering analysis (K-means, DBSCAN)
- Recommendation and ranking systems
- Graph analytics and centrality measures
