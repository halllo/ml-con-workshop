"""
Evaluation Component for Kubeflow Pipeline
Evaluates the trained recommendation model
"""

from kfp.dsl import component, Input, Output, Dataset, Model, Metrics


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.0.3", "numpy==1.24.3", "scikit-learn==1.3.2"]
)
def evaluate_model(
    test_data: Input[Dataset],
    model: Input[Model],
    metrics: Output[Metrics]
) -> str:
    """
    Evaluate recommendation model on test data

    Args:
        test_data: Input test dataset
        model: Input trained model
        metrics: Output evaluation metrics

    Returns:
        Evaluation status message
    """
    import pandas as pd
    import numpy as np
    import pickle
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load model
    logger.info(f"Loading model from {model.path}")
    with open(model.path, 'rb') as f:
        model_data = pickle.load(f)

    user_factors = model_data['user_factors']
    movie_factors = model_data['movie_factors']
    user_encoder = model_data['user_encoder']
    movie_encoder = model_data['movie_encoder']

    # Load test data
    logger.info(f"Loading test data from {test_data.path}")
    test_df = pd.read_csv(test_data.path)

    logger.info(f"Evaluating on {len(test_df)} ratings")

    # Make predictions on test set
    predictions = []
    actuals = []

    for _, row in test_df.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        actual_rating = row['rating']

        # Skip if user or movie not in training set
        if user_id not in user_encoder or movie_id not in movie_encoder:
            continue

        user_idx = user_encoder[user_id]
        movie_idx = movie_encoder[movie_id]

        # Predict rating
        predicted_rating = user_factors[user_idx] @ movie_factors[movie_idx]
        predicted_rating = np.clip(predicted_rating, 1.0, 5.0)

        predictions.append(predicted_rating)
        actuals.append(actual_rating)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)

    # Calculate coverage
    test_users = test_df['userId'].unique()
    test_movies = test_df['movieId'].unique()
    user_coverage = len(set(test_users) & set(user_encoder.keys())) / len(test_users)
    movie_coverage = len(set(test_movies) & set(movie_encoder.keys())) / len(test_movies)

    logger.info(f"Test RMSE: {rmse:.3f}")
    logger.info(f"Test MAE: {mae:.3f}")
    logger.info(f"User coverage: {user_coverage:.2%}")
    logger.info(f"Movie coverage: {movie_coverage:.2%}")

    # Log metrics to Kubeflow
    metrics.log_metric("test_rmse", float(rmse))
    metrics.log_metric("test_mae", float(mae))
    metrics.log_metric("user_coverage", float(user_coverage))
    metrics.log_metric("movie_coverage", float(movie_coverage))
    metrics.log_metric("n_test_ratings", len(test_df))
    metrics.log_metric("n_evaluated_ratings", len(predictions))

    # Return evaluation status
    status = f"Evaluation complete: RMSE={rmse:.3f}, MAE={mae:.3f}, Coverage={user_coverage:.2%}"
    logger.info(status)

    return status
