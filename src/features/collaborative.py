import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from typing import Tuple, Dict


def build_user_item_matrix(
    ratings: pl.DataFrame,
    user_col: str = "userId",
    item_col: str = "movieId",
    value_col: str = "rating"
) -> Tuple[csr_matrix, Dict[int, int], Dict[int, int]]:

    user_ids = ratings.get_column(user_col).to_numpy()
    item_ids = ratings.get_column(item_col).to_numpy()
    values = ratings.get_column(value_col).to_numpy()

    user_map = {u: i for i, u in enumerate(np.unique(user_ids))}
    item_map = {m: i for i, m in enumerate(np.unique(item_ids))}

    rows = np.array([user_map[u] for u in user_ids])
    cols = np.array([item_map[m] for m in item_ids])

    matrix = csr_matrix((values, (rows, cols)))

    return matrix, user_map, item_map


def run_svd_collaborative(
    ratings: pl.DataFrame,
    n_components: int = 50,
    user_col: str = "userId",
    item_col: str = "movieId",
    value_col: str = "rating",
    random_state: int = 42
):

    matrix, user_map, item_map = build_user_item_matrix(
        ratings,
        user_col=user_col,
        item_col=item_col,
        value_col=value_col
    )

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=random_state
    )

    user_embeddings = svd.fit_transform(matrix)
    item_embeddings = svd.components_.T

    user_df = pl.DataFrame(
        user_embeddings,
        schema=[f"cf_u_{i}" for i in range(n_components)]
    ).with_columns(
        pl.Series("userId", list(user_map.keys()))
    )

    item_df = pl.DataFrame(
        item_embeddings,
        schema=[f"cf_i_{i}" for i in range(n_components)]
    ).with_columns(
        pl.Series("movieId", list(item_map.keys()))
    )

    return user_df, item_df, svd