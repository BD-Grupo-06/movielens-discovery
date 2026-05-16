from pandas._libs import ops_dispatch
import numpy as np
import polars as pl
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy import sparse

def run_pca_embeddings(
    embeddings: np.ndarray,
    movie_ids: np.ndarray,
    n_components=0.9,
    random_state: int = 42
):
    pca = PCA(
        n_components=n_components,
        random_state=random_state
    )

    reduced = pca.fit_transform(embeddings)
    
    cols = [f"pc_{i}" for i in range(reduced.shape[1])]

    pca_df = pl.DataFrame(
        reduced,
        schema=cols
    ).with_columns(
        pl.Series("movieId", movie_ids)
    )

    variance_df = pl.DataFrame({
        "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)
    })

    return pca_df, variance_df, pca

def run_tsne(embedding: np.ndarray, random_state: int, perplexity: int = 30, max_samples: int = 5000):

    if embedding.shape[0] > max_samples:
        np.random.seed(random_state)
        indices = np.random.choice(embedding.shape[0], max_samples, replace=False)
        embedding_to_run = embedding[indices]
    else:
        embedding_to_run = embedding
        indices = np.arange(embedding.shape[0])

    tsne = TSNE(
        n_components=2, 
        random_state=random_state, 
        init="pca",
        perplexity=perplexity,
        n_jobs=-1
    )
    
    coords = tsne.fit_transform(embedding_to_run)
    
    return coords, indices
