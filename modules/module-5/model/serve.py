"""
Inference server for movie recommendation model
Compatible with KServe v2 protocol
"""

from typing import Dict, List
import logging
import os
from recommender import MovieRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommenderServer:
    """
    KServe-compatible inference server for movie recommendations
    """

    def __init__(self, model_path: str = "/mnt/models/model.pkl"):
        """
        Initialize server

        Args:
            model_path: Path to saved model
        """
        self.model_path = model_path
        self.model = None
        self.ready = False

    def load(self):
        """Load model from disk or S3"""
        try:
            logger.info(f"Loading model from {self.model_path}")

            # If path starts with /minio/, download from minio S3
            if self.model_path.startswith('/minio/'):
                logger.info("Detected minio path, downloading from S3...")
                local_path = self._download_from_minio(self.model_path)
            else:
                local_path = self.model_path

            self.model = MovieRecommender()
            self.model.load(local_path)
            self.ready = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _download_from_minio(self, minio_path: str) -> str:
        """Download model from minio S3"""
        import boto3
        from botocore.client import Config

        # Parse minio path: /minio/BUCKET/path/to/model
        parts = minio_path.strip('/').split('/', 2)
        if len(parts) < 3:
            raise ValueError(f"Invalid minio path: {minio_path}")

        bucket = parts[1]  # mlpipeline
        object_key = parts[2]  # path/to/model

        # Get minio credentials from environment or use defaults
        minio_endpoint = os.getenv('MINIO_ENDPOINT', 'minio-service.kubeflow:9000')
        minio_access_key = os.getenv('MINIO_ACCESS_KEY', 'minio')
        minio_secret_key = os.getenv('MINIO_SECRET_KEY', 'minio123')

        logger.info(f"Downloading from minio: bucket={bucket}, key={object_key}")

        # Create S3 client
        s3_client = boto3.client(
            's3',
            endpoint_url=f'http://{minio_endpoint}',
            aws_access_key_id=minio_access_key,
            aws_secret_access_key=minio_secret_key,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )

        # Download to local path
        local_path = '/tmp/model.pkl'
        s3_client.download_file(bucket, object_key, local_path)
        logger.info(f"Downloaded model to {local_path}")

        return local_path

    def predict(self, request: Dict) -> Dict:
        """
        Make predictions

        Request format:
        {
            "instances": [
                {"user_id": 1, "n_recommendations": 10},
                ...
            ]
        }

        Response format:
        {
            "predictions": [
                {
                    "user_id": 1,
                    "recommendations": [
                        {"movie_id": 123, "predicted_rating": 4.5},
                        ...
                    ]
                },
                ...
            ]
        }
        """
        if not self.ready:
            raise RuntimeError("Model not loaded")

        instances = request.get("instances", [])
        predictions = []

        for instance in instances:
            user_id = instance.get("user_id")
            n_recommendations = instance.get("n_recommendations", 10)
            genre_filter = instance.get("genre", None)

            if user_id is None:
                predictions.append({"error": "user_id is required"})
                continue

            # Get recommendations
            recommendations = self.model.recommend_movies(
                user_id=user_id,
                n_recommendations=n_recommendations,
                genre_filter=genre_filter
            )

            # Format response
            predictions.append({
                "user_id": user_id,
                "recommendations": recommendations  # Already formatted as dicts
            })

        return {"predictions": predictions}


# For KServe compatibility, create a simple HTTP server
if __name__ == "__main__":
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    # Get model path from environment
    model_path = os.getenv("MODEL_PATH", "/mnt/models/model.pkl")

    # Initialize server
    server = RecommenderServer(model_path=model_path)

    @app.route("/v1/models/recommender:predict", methods=["POST"])
    def predict():
        """Prediction endpoint"""
        try:
            request_data = request.get_json()
            response = server.predict(request_data)
            return jsonify(response)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/v1/models/recommender", methods=["GET"])
    def health():
        """Health check endpoint"""
        if server.ready:
            return jsonify({"name": "recommender", "ready": True})
        else:
            return jsonify({"name": "recommender", "ready": False}), 503

    # Load model on startup
    logger.info("Starting recommendation server")
    server.load()

    # Start server
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
