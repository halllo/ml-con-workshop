"""
Kubeflow Pipeline for Movie Recommendation System

This pipeline demonstrates end-to-end ML workflow:
1. Data Preparation - Download and split MovieLens dataset
2. Model Training - Train collaborative filtering model
3. Model Evaluation - Evaluate on test set
4. Model Deployment - Deploy to KServe
"""

from kfp import dsl, compiler
from kfp.dsl import pipeline
import sys
import os

# Import components
sys.path.append(os.path.dirname(__file__))
from components.data_prep_solution import prepare_data
from components.train_solution import train_model
from components.evaluate_solution import evaluate_model
from components.deploy_solution import deploy_model


@pipeline(
    name="movie-recommendation-pipeline",
    description="End-to-end pipeline for training and deploying movie recommendation model"
)
def recommendation_pipeline(
    dataset_size: str = "100k",
    test_ratio: float = 0.2,
    n_components: int = 20,
    random_state: int = 42,
    deploy_model_flag: bool = True,  # Set to True only if KServe is installed
    canary_traffic_percent: int = 0
):
    """
    Complete ML pipeline for movie recommendations

    Args:
        dataset_size: Size of MovieLens dataset (100k, 1m, etc.)
        test_ratio: Fraction of data for testing
        n_components: Number of latent factors for SVD
        random_state: Random seed for reproducibility
        deploy_model_flag: Whether to deploy the model
        canary_traffic_percent: Percentage for canary deployment (0-100)
    """

    # Step 1: Prepare data
    data_prep_task = prepare_data(
        dataset_size=dataset_size,
        test_ratio=test_ratio,
        random_state=random_state
    )
    data_prep_task.set_display_name("Prepare MovieLens Data")

    # Step 2: Train model
    train_task = train_model(
        train_data=data_prep_task.outputs["train_data"],
        movies_metadata=data_prep_task.outputs["movies_metadata"],
        n_components=n_components,
        random_state=random_state
    )
    train_task.set_display_name("Train Recommendation Model")
    train_task.after(data_prep_task)

    # Step 3: Evaluate model
    eval_task = evaluate_model(
        test_data=data_prep_task.outputs["test_data"],
        model=train_task.outputs["model"]
    )
    eval_task.set_display_name("Evaluate Model")
    eval_task.after(train_task)

    # Step 4: Deploy model (conditional)
    with dsl.Condition(deploy_model_flag == True):
        deploy_task = deploy_model(
            model=train_task.outputs["model"],
            service_name="movie-recommender",
            namespace="default",
            canary_traffic_percent=canary_traffic_percent
        )
        deploy_task.set_display_name("Deploy to KServe")
        deploy_task.after(eval_task)


def compile_pipeline(output_path: str = "recommendation_pipeline.yaml"):
    """
    Compile pipeline to YAML

    Args:
        output_path: Path to save compiled pipeline
    """
    compiler.Compiler().compile(
        pipeline_func=recommendation_pipeline,
        package_path=output_path
    )
    print(f"Pipeline compiled to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compile Kubeflow pipeline")
    parser.add_argument(
        "--output",
        type=str,
        default="recommendation_pipeline.yaml",
        help="Output path for compiled pipeline"
    )

    args = parser.parse_args()

    # Compile pipeline
    compile_pipeline(args.output)
