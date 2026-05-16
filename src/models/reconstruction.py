import numpy as np
import polars as pl
from scipy import sparse
from sklearn.preprocessing import StandardScaler

def reconstruction_error_pca(embeddings: np.ndarray, pca_model) -> float:
    transformed = pca_model.transform(embeddings)
    reconstructed = pca_model.inverse_transform(transformed)
    error = np.mean((embeddings - reconstructed) ** 2)
    return float(error)

def reconstruction_error_svd(
    ratings_df: pl.DataFrame, 
    user_df: pl.DataFrame, 
    item_df: pl.DataFrame
) -> float:
    df_pred = (
        ratings_df.select(["userId", "movieId", "rating"])
        .join(user_df, on="userId", how="inner")
        .join(item_df, on="movieId", how="inner", suffix="_item")
    )
    
    u_cols = [c for c in user_df.columns if c != "userId"]
    i_cols = [c for c in item_df.columns if c != "movieId"]
    
    U_k = df_pred.select(u_cols).to_numpy()
    I_k = df_pred.select(i_cols).to_numpy()
    
    ratings_pred = np.sum(U_k * I_k, axis=1)
    ratings_real = df_pred.get_column("rating").to_numpy()
    
    rmse = np.sqrt(np.mean((ratings_real - ratings_pred) ** 2))
    return float(rmse)
