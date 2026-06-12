"""Corrected genre-relevance evaluation for Week 10.

Recomputes the item-to-item genre evaluation from committed artifacts only
(no raw data needed), fixing the two self-referential metrics documented in
the Week 10 report:

- Recall@K: original denominator was the relevant count inside the system's
  own top-20 pool. Corrected denominator = relevant movies in the full
  evaluation universe (the 59,047 rated movies with genre labels).
- NDCG@K: original ideal DCG used the relevant count inside the retrieved
  top-K. Corrected ideal DCG assumes K relevant items, since every query has
  far more than K genre-relevant movies in the catalog.

It also evaluates the cluster-aware popularity hybrid (Week 7 segmentation
feeding the popularity ranking), which was built in
week10_baseline_recommender.ipynb but absent from the original metric table,
and emits the data-alignment table required for the hybrid claim.

Inputs (all committed in artifacts/):
    week10_popularity_global.csv    universe: movieId, genres, rating_count,
                                    bayesian_score, cluster
    week10_popularity_cluster.csv   per-cluster Bayesian ranking (cluster_rank)
    week10_content_recs_top20.parquet
    week10_svd_recs_top20.parquet
    week07_kmeans_assignments.csv
    week10_svd_item_factors.parquet

Outputs (artifacts/week10/):
    week10_genre_eval_corrected.csv
    week10_data_alignment.csv
    week10_evaluation_summary.json   (adds the genre_eval_corrected block)

Usage:
    python3 scripts/build_week10_eval_corrected.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_W10 = PROJECT_ROOT / "artifacts" / "week10"
ARTIFACTS_W07 = PROJECT_ROOT / "artifacts" / "week07"

K_VALUES = [5, 10, 20]
TOP_N = 20
NO_GENRES = "(no genres listed)"

LOG2 = np.log2(np.arange(2, TOP_N + 2))  # discounts for ranks 1..20
IDEAL_DCG_AT = {k: float((1.0 / LOG2[:k]).sum()) for k in K_VALUES}


def load_universe() -> pl.DataFrame:
    pop = pl.read_csv(ARTIFACTS_W10 / "week10_popularity_global.csv")
    return pop.with_columns(
        pl.when(pl.col("genres").is_null() | (pl.col("genres") == NO_GENRES))
        .then(pl.lit([], dtype=pl.List(pl.String)))
        .otherwise(pl.col("genres").str.split("|"))
        .alias("genres_list")
    )


def genre_matrix(universe: pl.DataFrame) -> tuple[np.ndarray, dict[int, int], list[str]]:
    """One-hot genre membership over the evaluation universe."""
    vocab = sorted(
        {g for row in universe["genres_list"].to_list() for g in row}
    )
    g_index = {g: i for i, g in enumerate(vocab)}
    onehot = np.zeros((universe.height, len(vocab)), dtype=bool)
    for r, row in enumerate(universe["genres_list"].to_list()):
        for g in row:
            onehot[r, g_index[g]] = True
    movie_row = {m: r for r, m in enumerate(universe["movieId"].to_list())}
    return onehot, movie_row, vocab


def recs_from_static_list(ranked_ids: list[int], queries: list[int]) -> dict[int, list[int]]:
    """Same global list for every query, excluding the query itself."""
    head = ranked_ids[: TOP_N + 1]
    return {q: [m for m in head if m != q][:TOP_N] for q in queries}


def recs_from_cluster_list(
    cluster_table: pl.DataFrame, query_cluster: dict[int, int], queries: list[int]
) -> dict[int, list[int]]:
    by_cluster: dict[int, list[int]] = {
        c: grp.sort("cluster_rank")["movieId"].to_list()[: TOP_N + 1]
        for (c,), grp in cluster_table.group_by("cluster")
    }
    return {
        q: [m for m in by_cluster[query_cluster[q]] if m != q][:TOP_N]
        for q in queries
    }


def recs_from_parquet(path: Path) -> dict[int, list[int]]:
    df = pl.read_parquet(path).sort(["query_movieId", "rank"])
    return {
        int(q): grp["rec_movieId"].to_list()
        for (q,), grp in df.group_by("query_movieId", maintain_order=True)
    }


def evaluate(
    recs: dict[int, list[int]],
    queries: list[int],
    onehot: np.ndarray,
    movie_row: dict[int, int],
    n_relevant: dict[int, int],
) -> dict[int, dict[str, float]]:
    """Per-K means of precision, pool recall (legacy), catalog recall,
    legacy NDCG, corrected NDCG, and hit rate."""
    acc = {
        k: {key: 0.0 for key in (
            "precision", "pool_recall", "catalog_recall",
            "ndcg_legacy", "ndcg", "hit_rate",
        )}
        for k in K_VALUES
    }
    for q in queries:
        q_vec = onehot[movie_row[q]]
        flags = np.zeros(TOP_N, dtype=bool)
        rec_ids = recs[q]
        for i, m in enumerate(rec_ids):
            row = movie_row.get(m)
            if row is not None and (onehot[row] & q_vec).any():
                flags[i] = True
        pool_relevant = int(flags.sum())
        gains = flags / LOG2
        for k in K_VALUES:
            hits = int(flags[:k].sum())
            dcg = float(gains[:k].sum())
            # Legacy ideal DCG: built from the relevant count inside top-K
            # (the self-referential definition being corrected).
            idcg_legacy = float((1.0 / LOG2[:hits]).sum()) if hits else 0.0
            acc[k]["precision"] += hits / k
            acc[k]["pool_recall"] += hits / pool_relevant if pool_relevant else 0.0
            acc[k]["catalog_recall"] += hits / n_relevant[q]
            acc[k]["ndcg_legacy"] += dcg / idcg_legacy if idcg_legacy else 0.0
            acc[k]["ndcg"] += dcg / IDEAL_DCG_AT[k]
            acc[k]["hit_rate"] += 1.0 if hits else 0.0
    n = len(queries)
    return {k: {key: v / n for key, v in vals.items()} for k, vals in acc.items()}


def main() -> None:
    universe = load_universe()
    onehot, movie_row, vocab = genre_matrix(universe)
    print(f"Universe: {universe.height:,} rated movies, {len(vocab)} genres")

    content_recs = recs_from_parquet(ARTIFACTS_W10 / "week10_content_recs_top20.parquet")
    svd_recs = recs_from_parquet(ARTIFACTS_W10 / "week10_svd_recs_top20.parquet")

    # Query set: intersection of both systems' query sets, restricted to
    # movies with at least one genre label (same rule as the notebook).
    has_genre = {
        m for m, row in movie_row.items() if onehot[row].any()
    }
    queries = sorted(set(content_recs) & set(svd_recs) & has_genre)
    print(f"Queries: {len(queries):,} "
          f"(content {len(content_recs):,} ∩ svd {len(svd_recs):,} ∩ with-genres)")

    # Relevant-universe size per query: movies sharing >= 1 genre, minus the
    # query itself. Vectorized: (universe one-hot) @ (query genres) > 0.
    q_rows = np.array([movie_row[q] for q in queries])
    share_counts = (onehot.astype(np.int32) @ onehot[q_rows].T.astype(np.int32) > 0).sum(axis=0)
    n_relevant = {q: int(c) - 1 for q, c in zip(queries, share_counts)}
    mean_rel = float(np.mean([n_relevant[q] for q in queries]))
    print(f"Mean genre-relevant movies per query: {mean_rel:,.0f}")

    # Popularity (global): static top-20 by raw rating count — exactly the
    # list the original evaluation used (see report Section 7.4).
    by_count = universe.sort("rating_count", descending=True)["movieId"].to_list()
    popularity_recs = recs_from_static_list(by_count, queries)

    # Cluster-aware popularity hybrid: Bayesian ranking inside the query's
    # Week 7 cluster.
    cluster_table = pl.read_csv(ARTIFACTS_W10 / "week10_popularity_cluster.csv")
    query_cluster = dict(
        universe.filter(pl.col("movieId").is_in(queries))
        .select("movieId", "cluster").iter_rows()
    )
    cluster_recs = recs_from_cluster_list(cluster_table, query_cluster, queries)

    systems = {
        "popularity_global": popularity_recs,
        "cluster_popularity": cluster_recs,
        "content_cosine": content_recs,
        "svd_collaborative": svd_recs,
    }
    results = {
        name: evaluate(recs, queries, onehot, movie_row, n_relevant)
        for name, recs in systems.items()
    }

    # Reproduction check: legacy metrics must match the original run
    # (week10_evaluation_results.csv) for the three original systems.
    original = pl.read_csv(ARTIFACTS_W10 / "week10_evaluation_results.csv")
    worst = 0.0
    for row in original.iter_rows(named=True):
        got = results[row["system"]][row["k"]]
        for ours, theirs in (
            ("precision", "precision_at_k"), ("pool_recall", "recall_at_k"),
            ("ndcg_legacy", "ndcg_at_k"), ("hit_rate", "hit_rate_at_k"),
        ):
            worst = max(worst, abs(got[ours] - row[theirs]))
    print(f"Reproduction check vs original CSV: max abs diff = {worst:.4f}")
    if worst > 0.002:
        raise SystemExit("Legacy metrics do not reproduce the original run.")

    out_rows = [
        {
            "system": name, "k": k,
            "precision_at_k": round(m["precision"], 4),
            "pool_recall_at_k_legacy": round(m["pool_recall"], 4),
            "catalog_recall_at_k": round(m["catalog_recall"], 6),
            "ndcg_at_k_legacy": round(m["ndcg_legacy"], 4),
            "ndcg_at_k_corrected": round(m["ndcg"], 4),
            "hit_rate_at_k": round(m["hit_rate"], 4),
            "n_queries": len(queries),
        }
        for name, per_k in results.items() for k, m in per_k.items()
    ]
    eval_path = ARTIFACTS_W10 / "week10_genre_eval_corrected.csv"
    pl.DataFrame(out_rows).write_csv(eval_path)
    print(f"Wrote {eval_path}")

    # Data-alignment table for the hybrid claim: how each layer's keys map
    # onto the next, with coverage counts on movieId.
    assignments = pl.read_csv(ARTIFACTS_W07 / "week07_kmeans_assignments.csv")
    svd_factors = pl.read_parquet(ARTIFACTS_W10 / "week10_svd_item_factors.parquet")
    alignment = pl.DataFrame(
        {
            "layer": [
                "movies_catalog (week 3)",
                "autoencoder embeddings AE-13 (week 7)",
                "k-means assignments k=7 (week 7)",
                "rated movies with genres = evaluation universe (week 10)",
                "cluster popularity ranking (week 10 hybrid)",
                "SVD item factors (week 10)",
                "query set content/svd (week 10)",
                "evaluated queries (intersection, with genres)",
            ],
            "join_key": ["movieId"] * 8,
            "n_movies": [
                62423,
                62423,
                assignments.height,
                universe.height,
                cluster_table.height,
                svd_factors.height,
                len(content_recs),
                len(queries),
            ],
            "coverage_of_catalog_pct": [
                round(100 * n / 62423, 2)
                for n in (
                    62423, 62423, assignments.height, universe.height,
                    cluster_table.height, svd_factors.height,
                    len(content_recs), len(queries),
                )
            ],
        }
    )
    align_path = ARTIFACTS_W10 / "week10_data_alignment.csv"
    alignment.write_csv(align_path)
    print(f"Wrote {align_path}")

    summary_path = ARTIFACTS_W10 / "week10_evaluation_summary.json"
    summary = json.loads(summary_path.read_text())
    corrected_block = {
        "evaluation_type": "item_to_item_genre_relevance_corrected",
        "n_query_movies": len(queries),
        "universe_size": universe.height,
        "mean_relevant_per_query": round(mean_rel),
        "recall_note": (
            "catalog_recall_at_k usa como denominador el numero de peliculas "
            "del universo (59,047) que comparten al menos un genero con la "
            "query (en promedio ~{:,.0f}), en lugar del pool propio del "
            "sistema. Por eso los valores son pequenos por construccion: con "
            "miles de items relevantes, Recall@20 <= 20/relevantes. "
            "Precision@K y NDCG@K son las metricas informativas en este "
            "protocolo; ndcg_at_k_corrected normaliza contra el DCG ideal de "
            "K items relevantes."
        ).format(mean_rel),
        "results": {
            name: {
                f"k_{k}": {
                    "precision": round(m["precision"], 6),
                    "catalog_recall": round(m["catalog_recall"], 6),
                    "ndcg_corrected": round(m["ndcg"], 6),
                    "hit_rate": round(m["hit_rate"], 6),
                }
                for k, m in per_k.items()
            }
            for name, per_k in results.items()
        },
        "artifacts": [
            "week10_genre_eval_corrected.csv",
            "week10_data_alignment.csv",
        ],
    }
    rebuilt = {}
    for key, value in summary.items():
        if key == "genre_eval_corrected":
            continue
        rebuilt[key] = value
        if key == "results":
            rebuilt["genre_eval_corrected"] = corrected_block
    summary_path.write_text(json.dumps(rebuilt, indent=2, ensure_ascii=False) + "\n")
    print(f"Updated {summary_path}")


if __name__ == "__main__":
    main()
