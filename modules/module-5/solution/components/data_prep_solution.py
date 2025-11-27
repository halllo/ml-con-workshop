"""
Data Preparation Component for Kubeflow Pipeline
Downloads and prepares MovieLens dataset
"""

from kfp.dsl import component, Output, Dataset


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.0.3", "scikit-learn==1.3.2"]
)
def prepare_data(
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    movies_metadata: Output[Dataset],
    dataset_size: str = "100k",
    test_ratio: float = 0.2,
    random_state: int = 42
):
    """
    Download and prepare MovieLens dataset

    Args:
        train_data: Output training dataset
        test_data: Output test dataset
        dataset_size: Size of dataset to download (100k, 1m, etc.)
        test_ratio: Fraction of data for test set
        random_state: Random seed for reproducibility
    """
    import urllib.request
    import zipfile
    import pandas as pd
    import os
    from sklearn.model_selection import train_test_split
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Download MovieLens 100K dataset
    dataset_url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = "/tmp/ml-100k.zip"

    logger.info(f"Downloading MovieLens dataset from {dataset_url}")
    urllib.request.urlretrieve(dataset_url, zip_path)

    # Extract
    logger.info("Extracting dataset")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("/tmp")

    # Load ratings
    logger.info("Loading ratings data")
    ratings = pd.read_csv(
        "/tmp/ml-100k/u.data",
        sep='\t',
        names=['userId', 'movieId', 'rating', 'timestamp'],
        engine='python'
    )

    logger.info(f"Loaded {len(ratings)} ratings")
    logger.info(f"Users: {ratings['userId'].nunique()}")
    logger.info(f"Movies: {ratings['movieId'].nunique()}")

    # Create train/test split
    train_df, test_df = train_test_split(
        ratings,
        test_size=test_ratio,
        random_state=random_state
    )

    logger.info(f"Train set: {len(train_df)} ratings")
    logger.info(f"Test set: {len(test_df)} ratings")

    # Load movie metadata
    logger.info("Loading movie metadata")
    movies = pd.read_csv(
        "/tmp/ml-100k/u.item",
        sep='|',
        names=['movieId', 'title', 'release_date', 'video_release_date', 'imdb_url',
               'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
               'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
               'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
        encoding='latin-1',
        engine='python'
    )

    # Extract genre columns
    genre_columns = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    # Create genres list for each movie
    movies['genres'] = movies[genre_columns].apply(
        lambda row: [genre for genre, val in zip(genre_columns, row) if val == 1],
        axis=1
    )

    # Keep only necessary columns
    movies_df = movies[['movieId', 'title', 'genres']]

    logger.info(f"Loaded {len(movies_df)} movies")

    # Save to output paths
    train_df.to_csv(train_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)
    movies_df.to_csv(movies_metadata.path, index=False)

    logger.info(f"Data preparation complete")
