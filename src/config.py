from pathlib import Path

# Common constants
DEFAULT_RANDOM_STATE = 42

# URLs
MOVIE_LENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"

# Paths
DATASET_DIR_NAME = "ml-25m"
ARCHIVE_NAME = "ml-25m.zip"

# Raw table names to filenames
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
