"""
Training Component for Kubeflow Pipeline
Trains the movie recommendation model
"""

from kfp.dsl import component, Input, Output, Dataset, Model, Metrics


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.0.3", "numpy==1.24.3", "scikit-learn==1.3.2"]
)
def train_model(
    train_data: Input[Dataset],
    movies_metadata: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    n_components: int = 20,
    random_state: int = 42
):
    """
    Train recommendation model using collaborative filtering

    Args:
        train_data: Input training dataset
        model: Output trained model
        metrics: Output training metrics
        n_components: Number of latent factors for SVD
        random_state: Random seed
    """
    import pandas as pd
    import numpy as np
    import pickle
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics import mean_squared_error
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load training data
    logger.info(f"Loading training data from {train_data.path}")
    ratings_df = pd.read_csv(train_data.path)

    logger.info(f"Training on {len(ratings_df)} ratings")

    # Load movie metadata
    logger.info(f"Loading movie metadata from {movies_metadata.path}")
    movies_df = pd.read_csv(movies_metadata.path)

    # Parse genres from string representation back to list
    import ast
    movies_df['genres'] = movies_df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Create movie metadata dictionary
    movie_metadata = {}
    for _, row in movies_df.iterrows():
        movie_metadata[row['movieId']] = {
            'title': row['title'],
            'genres': row['genres']
        }

    logger.info(f"Loaded metadata for {len(movie_metadata)} movies")

    # Create encoders
    unique_users = ratings_df['userId'].unique()
    unique_movies = ratings_df['movieId'].unique()

    user_encoder = {user_id: idx for idx, user_id in enumerate(unique_users)}
    movie_encoder = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}

    logger.info(f"Encoded {len(unique_users)} users and {len(unique_movies)} movies")

    # Create user-movie matrix
    n_users = len(user_encoder)
    n_movies = len(movie_encoder)
    user_movie_matrix = np.zeros((n_users, n_movies))

    for _, row in ratings_df.iterrows():
        user_idx = user_encoder[row['userId']]
        movie_idx = movie_encoder[row['movieId']]
        user_movie_matrix[user_idx, movie_idx] = row['rating']

    logger.info(f"Created user-movie matrix: {user_movie_matrix.shape}")

    # Train SVD model
    logger.info(f"Training SVD model with {n_components} components")
    svd_model = TruncatedSVD(n_components=n_components, random_state=random_state)
    user_factors = svd_model.fit_transform(user_movie_matrix)
    movie_factors = svd_model.components_.T

    # Calculate training metrics
    predicted_ratings = user_factors @ movie_factors.T

    # RMSE on observed ratings
    observed_mask = user_movie_matrix > 0
    observed_ratings = user_movie_matrix[observed_mask]
    predicted_observed = predicted_ratings[observed_mask]

    rmse = np.sqrt(mean_squared_error(observed_ratings, predicted_observed))
    explained_variance = svd_model.explained_variance_ratio_.sum()

    logger.info(f"Training complete. RMSE: {rmse:.3f}, "
               f"Explained Variance: {explained_variance:.3f}")

    # Log metrics to Kubeflow
    metrics.log_metric("rmse", float(rmse))
    metrics.log_metric("explained_variance", float(explained_variance))
    metrics.log_metric("n_components", n_components)
    metrics.log_metric("n_users", len(unique_users))
    metrics.log_metric("n_movies", len(unique_movies))
    metrics.log_metric("n_ratings", len(ratings_df))

    # Save model
    model_data = {
        'user_factors': user_factors,
        'movie_factors': movie_factors,
        'user_encoder': user_encoder,
        'movie_encoder': movie_encoder,
        'user_decoder': {idx: user_id for user_id, idx in user_encoder.items()},
        'movie_decoder': {idx: movie_id for movie_id, idx in movie_encoder.items()},
        'user_movie_matrix': user_movie_matrix,
        'movie_metadata': movie_metadata,
        'n_components': n_components,
        'random_state': random_state
    }

    with open(model.path, 'wb') as f:
        pickle.dump(model_data, f)

    logger.info(f"Model saved to {model.path}")
