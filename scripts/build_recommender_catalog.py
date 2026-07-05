"""Build the movie catalog + recommendations JSON consumed by the web recommender page.

Reuses the Week 10 precomputed recommendation artifacts (content_cosine and svd_collaborative,
each evaluated over 5,001 query movies in the Week 10 offline evaluation) rather than computing
anything new. No new modeling: this is a read-only projection of already-produced, already-defended
artifacts into a shape the web UI can fetch directly.

Usage:
    env/bin/python scripts/build_recommender_catalog.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TOP_K = 10

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "week03_v1"
WEEK10_DIR = PROJECT_ROOT / "artifacts" / "week10"
OUT_PATH = WEEK10_DIR / "week10_recommender_catalog.json"


def imdb_url(imdb_id) -> str | None:
    if imdb_id is None or (isinstance(imdb_id, float) and np.isnan(imdb_id)):
        return None
    return f"https://www.imdb.com/title/tt{int(imdb_id):07d}/"


def tmdb_url(tmdb_id) -> str | None:
    if tmdb_id is None or (isinstance(tmdb_id, float) and np.isnan(tmdb_id)):
        return None
    return f"https://www.themoviedb.org/movie/{int(tmdb_id)}"


def main() -> None:
    catalog = pd.read_parquet(DATA_DIR / "movies_catalog.parquet")
    popularity = pd.read_csv(WEEK10_DIR / "week10_popularity_global.csv")
    content_recs = pd.read_parquet(WEEK10_DIR / "week10_content_recs_top20.parquet")
    svd_recs = pd.read_parquet(WEEK10_DIR / "week10_svd_recs_top20.parquet")

    title_map = dict(zip(catalog["movieId"], catalog["title"]))
    genres_map = dict(zip(catalog["movieId"], catalog["genres"]))
    imdb_map = dict(zip(catalog["movieId"], catalog["imdbId"]))
    tmdb_map = dict(zip(catalog["movieId"], catalog["tmdbId"]))
    rating_count_map = dict(zip(popularity["movieId"], popularity["rating_count"]))

    query_ids = sorted(set(content_recs["query_movieId"]) | set(svd_recs["query_movieId"]))

    systems = {
        "content_cosine": (content_recs, "cosine_similarity"),
        "svd_collaborative": (svd_recs, "svd_score"),
    }

    recommendations: dict[str, dict[str, list[dict]]] = {}
    all_movie_ids: set[int] = set(query_ids)

    for system_name, (df, score_col) in systems.items():
        top = df[df["rank"] <= TOP_K]
        by_query: dict[str, list[dict]] = {}
        for movie_id, group in top.groupby("query_movieId"):
            group = group.sort_values("rank")
            entries = []
            for _, row in group.iterrows():
                rec_id = int(row["rec_movieId"])
                all_movie_ids.add(rec_id)
                entries.append({"movieId": rec_id, "score": round(float(row[score_col]), 4)})
            by_query[str(int(movie_id))] = entries
        recommendations[system_name] = by_query

    movies = []
    for movie_id in sorted(all_movie_ids):
        tmdb_id = tmdb_map.get(movie_id)
        movies.append({
            "movieId": int(movie_id),
            "title": title_map.get(movie_id, str(movie_id)),
            "genres": (genres_map.get(movie_id) or "").split("|"),
            "imdbId": int(imdb_map[movie_id]) if movie_id in imdb_map and not pd.isna(imdb_map[movie_id]) else None,
            "tmdbId": int(tmdb_id) if tmdb_id is not None and not pd.isna(tmdb_id) else None,
            "imdbUrl": imdb_url(imdb_map.get(movie_id)),
            "tmdbUrl": tmdb_url(tmdb_id),
            "posterUrl": None,
            "ratingCount": int(rating_count_map.get(movie_id, 0)),
        })

    payload = {
        "meta": {
            "generated_from": "artifacts/week10 (week10_content_recs_top20.parquet, week10_svd_recs_top20.parquet)",
            "systems": list(systems.keys()),
            "top_k": TOP_K,
            "n_query_movies": len(query_ids),
            "n_movies": len(movies),
        },
        "movies": movies,
        "queryMovieIds": [int(i) for i in query_ids],
        "recommendations": recommendations,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH} ({len(movies):,} movies, {len(query_ids):,} query movies)")


if __name__ == "__main__":
    main()
