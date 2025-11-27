"""
Movie Recommendation Model using Collaborative Filtering
Simple implementation for Kubeflow Pipeline demonstration
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieRecommender:
    """
    Collaborative Filtering recommender using Matrix Factorization (SVD)
    """

    def __init__(self, n_components: int = 20, random_state: int = 42):
        """
        Initialize recommender

        Args:
            n_components: Number of latent factors
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.model = TruncatedSVD(
            n_components=n_components,
            random_state=random_state
        )
        self.user_encoder = {}
        self.movie_encoder = {}
        self.user_decoder = {}
        self.movie_decoder = {}
        self.user_movie_matrix = None
        self.user_factors = None
        self.movie_factors = None
        self.movie_metadata = {}  # Store movie names and genres

    def _create_encoders(self, ratings_df: pd.DataFrame):
        """Create user and movie ID encoders"""
        unique_users = ratings_df['userId'].unique()
        unique_movies = ratings_df['movieId'].unique()

        self.user_encoder = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.movie_encoder = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
        self.user_decoder = {idx: user_id for user_id, idx in self.user_encoder.items()}
        self.movie_decoder = {idx: movie_id for movie_id, idx in self.movie_encoder.items()}

        logger.info(f"Encoded {len(unique_users)} users and {len(unique_movies)} movies")

    def _create_user_movie_matrix(self, ratings_df: pd.DataFrame) -> np.ndarray:
        """Create user-movie rating matrix"""
        n_users = len(self.user_encoder)
        n_movies = len(self.movie_encoder)

        matrix = np.zeros((n_users, n_movies))

        for _, row in ratings_df.iterrows():
            user_idx = self.user_encoder[row['userId']]
            movie_idx = self.movie_encoder[row['movieId']]
            matrix[user_idx, movie_idx] = row['rating']

        logger.info(f"Created user-movie matrix: {matrix.shape}")
        logger.info(f"Matrix sparsity: {(matrix == 0).sum() / matrix.size * 100:.2f}%")

        return matrix

    def train(self, ratings_df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the recommender model

        Args:
            ratings_df: DataFrame with columns [userId, movieId, rating]

        Returns:
            Training metrics
        """
        logger.info(f"Training on {len(ratings_df)} ratings")

        # Create encoders
        self._create_encoders(ratings_df)

        # Create user-movie matrix
        self.user_movie_matrix = self._create_user_movie_matrix(ratings_df)

        # Train SVD model
        self.user_factors = self.model.fit_transform(self.user_movie_matrix)
        self.movie_factors = self.model.components_.T

        # Calculate training metrics
        predicted_ratings = self.user_factors @ self.movie_factors.T

        # Get RMSE on observed ratings
        observed_mask = self.user_movie_matrix > 0
        observed_ratings = self.user_movie_matrix[observed_mask]
        predicted_observed = predicted_ratings[observed_mask]

        rmse = np.sqrt(mean_squared_error(observed_ratings, predicted_observed))
        explained_variance = self.model.explained_variance_ratio_.sum()

        metrics = {
            'rmse': float(rmse),
            'explained_variance': float(explained_variance),
            'n_components': self.n_components,
            'n_users': len(self.user_encoder),
            'n_movies': len(self.movie_encoder),
            'n_ratings': len(ratings_df)
        }

        logger.info(f"Training complete. RMSE: {rmse:.3f}, "
                   f"Explained Variance: {explained_variance:.3f}")

        return metrics

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for a user-movie pair

        Args:
            user_id: User ID
            movie_id: Movie ID

        Returns:
            Predicted rating (1-5 scale)
        """
        if user_id not in self.user_encoder:
            # Unknown user - return average rating
            return 3.5

        if movie_id not in self.movie_encoder:
            # Unknown movie - return average rating
            return 3.5

        user_idx = self.user_encoder[user_id]
        movie_idx = self.movie_encoder[movie_id]

        rating = self.user_factors[user_idx] @ self.movie_factors[movie_idx]

        # Clip to valid rating range
        return float(np.clip(rating, 1.0, 5.0))

    def recommend_movies(
        self,
        user_id: int,
        n_recommendations: int = 10,
        exclude_watched: bool = True,
        genre_filter: str = None
    ) -> List[Dict]:
        """
        Get movie recommendations for a user

        Args:
            user_id: User ID
            n_recommendations: Number of movies to recommend
            exclude_watched: Whether to exclude already watched movies
            genre_filter: Optional genre to filter by (e.g., "Action", "Comedy")

        Returns:
            List of recommendation dictionaries with movie_id, movie_name, score, genres
        """
        if user_id not in self.user_encoder:
            logger.warning(f"User {user_id} not found, returning empty recommendations")
            return []

        user_idx = self.user_encoder[user_id]

        # Get predictions for all movies
        user_predictions = self.user_factors[user_idx] @ self.movie_factors.T

        # Clip to valid range
        user_predictions = np.clip(user_predictions, 1.0, 5.0)

        # Exclude watched movies if requested
        if exclude_watched:
            watched_mask = self.user_movie_matrix[user_idx] > 0
            user_predictions[watched_mask] = -1  # Mark as already watched

        # Get top N movie indices (get more if we need to filter by genre)
        fetch_count = n_recommendations * 5 if genre_filter else n_recommendations
        top_indices = np.argsort(user_predictions)[::-1][:fetch_count]

        # Convert to recommendation dictionaries
        recommendations = []
        for idx in top_indices:
            if user_predictions[idx] <= 0:  # Skip watched movies
                continue

            movie_id = int(self.movie_decoder[idx])
            score = float(user_predictions[idx])

            # Get movie metadata
            metadata = self.movie_metadata.get(movie_id, {})
            movie_name = metadata.get('title', f'Movie {movie_id}')
            genres = metadata.get('genres', [])

            # Apply genre filter if specified
            if genre_filter:
                if not any(genre_filter.lower() in g.lower() for g in genres):
                    continue

            recommendations.append({
                'movie_id': movie_id,
                'movie_name': movie_name,
                'score': score,
                'genres': genres
            })

            if len(recommendations) >= n_recommendations:
                break

        return recommendations[:n_recommendations]

    def save(self, filepath: str):
        """Save model to disk"""
        model_data = {
            'user_factors': self.user_factors,
            'movie_factors': self.movie_factors,
            'user_encoder': self.user_encoder,
            'movie_encoder': self.movie_encoder,
            'user_decoder': self.user_decoder,
            'movie_decoder': self.movie_decoder,
            'user_movie_matrix': self.user_movie_matrix,
            'movie_metadata': self.movie_metadata,
            'n_components': self.n_components,
            'random_state': self.random_state
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.user_factors = model_data['user_factors']
        self.movie_factors = model_data['movie_factors']
        self.movie_metadata = model_data.get('movie_metadata', {})
        self.user_encoder = model_data['user_encoder']
        self.movie_encoder = model_data['movie_encoder']
        self.user_decoder = model_data['user_decoder']
        self.movie_decoder = model_data['movie_decoder']
        self.user_movie_matrix = model_data['user_movie_matrix']
        self.n_components = model_data['n_components']
        self.random_state = model_data['random_state']

        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 3:
        print("Usage: python recommender.py <train_data.csv> <output_model.pkl>")
        sys.exit(1)

    train_file = sys.argv[1]
    output_file = sys.argv[2]

    # Load data
    logger.info(f"Loading training data from {train_file}")
    ratings = pd.read_csv(train_file)

    # Train model
    recommender = MovieRecommender(n_components=20)
    metrics = recommender.train(ratings)

    logger.info(f"Training metrics: {metrics}")

    # Save model
    recommender.save(output_file)

    # Test recommendations
    test_user = ratings['userId'].iloc[0]
    recommendations = recommender.recommend_movies(test_user, n_recommendations=5)

    logger.info(f"\nTop 5 recommendations for user {test_user}:")
    for movie_id, rating in recommendations:
        logger.info(f"  Movie {movie_id}: predicted rating {rating:.2f}")
