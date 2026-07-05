"""Fetch TMDb poster URLs for the sampled movies used by the Week 12 3D graph viewer.

Only the sampled subset of movies in `artifacts/week12/movie_graph_viz.json` is looked up
(a few hundred, not the full 62k-movie catalog), since posters are only needed for what the
3D visualization actually renders.

Usage:
    export TMDB_API_KEY=your_v3_api_key
    env/bin/python scripts/fetch_posters.py

Requires a free TMDb API key: https://www.themoviedb.org/settings/api
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

DEFAULT_GRAPH_JSON = "artifacts/week12/movie_graph_viz.json"
DEFAULT_CACHE_JSON = "artifacts/week12/poster_urls.json"
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{tmdb_id}"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/{size}"


def load_api_key(explicit_key: str | None, project_root: Path) -> str:
    if explicit_key:
        return explicit_key
    import os

    if os.environ.get("TMDB_API_KEY"):
        return os.environ["TMDB_API_KEY"]

    env_file = project_root / "env" / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("TMDB_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    raise SystemExit(
        "No TMDb API key found. Set TMDB_API_KEY as an environment variable, pass "
        "--api-key, or put TMDB_API_KEY=... in env/.env. Get a free key at "
        "https://www.themoviedb.org/settings/api"
    )


def load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}


def fetch_movie_details(tmdb_id: int, api_key: str, session: requests.Session) -> dict | None:
    resp = session.get(
        TMDB_MOVIE_URL.format(tmdb_id=tmdb_id),
        params={"api_key": api_key},
        timeout=10,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    data = resp.json()
    return {"poster_path": data.get("poster_path"), "overview": data.get("overview") or None}


def fetch_posters(
    graph_json_path: Path,
    cache_json_path: Path,
    api_key: str,
    poster_size: str,
    sleep_seconds: float,
    nodes_key: str = "nodes",
) -> dict:
    payload = json.loads(graph_json_path.read_text())
    nodes = payload[nodes_key]
    cache = load_cache(cache_json_path)

    session = requests.Session()
    n_fetched, n_cached, n_missing, n_skipped = 0, 0, 0, 0

    for i, node in enumerate(nodes):
        movie_id = str(node["movieId"])
        tmdb_id = node.get("tmdbId")

        # "overview" was added after the first pass, so entries cached before that
        # (poster_path/poster_url only) are re-fetched once to backfill it.
        if movie_id in cache and "overview" in cache[movie_id]:
            n_cached += 1
            continue
        if tmdb_id is None:
            cache[movie_id] = {"poster_path": None, "poster_url": None, "overview": None}
            n_skipped += 1
            continue

        try:
            details = fetch_movie_details(tmdb_id, api_key, session)
        except requests.RequestException as exc:
            print(f"  warning: failed to fetch tmdbId={tmdb_id} (movieId={movie_id}): {exc}")
            details = None

        poster_path = details.get("poster_path") if details else None
        overview = details.get("overview") if details else None
        poster_url = (
            TMDB_IMAGE_BASE.format(size=poster_size) + poster_path if poster_path else None
        )
        cache[movie_id] = {"poster_path": poster_path, "poster_url": poster_url, "overview": overview}
        if poster_path:
            n_fetched += 1
        else:
            n_missing += 1
        time.sleep(sleep_seconds)

        if (i + 1) % 200 == 0:
            cache_json_path.parent.mkdir(parents=True, exist_ok=True)
            cache_json_path.write_text(json.dumps(cache, indent=2))

    cache_json_path.parent.mkdir(parents=True, exist_ok=True)
    cache_json_path.write_text(json.dumps(cache, indent=2))

    print(
        f"Poster cache updated: {n_fetched} newly fetched, {n_cached} already cached, "
        f"{n_missing} not found on TMDb, {n_skipped} skipped (no tmdbId). "
        f"Cache: {cache_json_path}"
    )

    for node in nodes:
        entry = cache.get(str(node["movieId"]), {})
        node["posterUrl"] = entry.get("poster_url")
        node["overview"] = entry.get("overview")

    graph_json_path.write_text(json.dumps(payload, indent=2))
    print(f"Merged posterUrl/overview into {graph_json_path}")

    return cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch TMDb poster URLs for the sampled Week 12 movie graph and merge them "
        "into movie_graph_viz.json."
    )
    parser.add_argument("--graph-json", default=DEFAULT_GRAPH_JSON)
    parser.add_argument("--cache-json", default=DEFAULT_CACHE_JSON)
    parser.add_argument("--api-key", default=None, help="TMDb v3 API key (overrides env/file).")
    parser.add_argument("--poster-size", default="w342", help="TMDb poster size, e.g. w185/w342/w500.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Delay between requests, in seconds.")
    parser.add_argument("--nodes-key", default="nodes", help="Top-level JSON key holding the list of movie records.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    api_key = load_api_key(args.api_key, project_root)
    fetch_posters(
        graph_json_path=project_root / args.graph_json,
        cache_json_path=project_root / args.cache_json,
        api_key=api_key,
        poster_size=args.poster_size,
        sleep_seconds=args.sleep,
        nodes_key=args.nodes_key,
    )


if __name__ == "__main__":
    main()
