"""
Deployment Component for Kubeflow Pipeline
Deploys model to KServe
"""

from kfp.dsl import component, Input, Model


@component(
    base_image="python:3.11-slim",
    packages_to_install=["kubernetes==28.1.0"]
)
def deploy_model(
    model: Input[Model],
    service_name: str = "movie-recommender",
    namespace: str = "default",
    canary_traffic_percent: int = 0
) -> str:
    """
    Deploy model to KServe

    Args:
        model: Input trained model
        service_name: Name for the KServe InferenceService
        namespace: Kubernetes namespace
        canary_traffic_percent: Percentage of traffic for canary (0-100)

    Returns:
        Deployment status message
    """
    from kubernetes import client, config
    import logging
    import json

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load Kubernetes config
    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()

    # Create KServe InferenceService manifest
    inference_service = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": service_name,
            "namespace": namespace
        },
        "spec": {
            "predictor": {
                "containers": [
                    {
                        "name": "kserve-container",
                        "image": "kind.local/movie-recommender:latest",
                        "imagePullPolicy": "IfNotPresent",
                        "env": [
                            {
                                "name": "MODEL_PATH",
                                "value": model.path
                            }
                        ],
                        "resources": {
                            "requests": {
                                "memory": "1Gi",
                                "cpu": "1000m"
                            },
                            "limits": {
                                "memory": "1Gi",
                                "cpu": "1000m"
                            }
                        }
                    }
                ]
            }
        }
    }

    # Add canary configuration if specified
    if canary_traffic_percent > 0:
        inference_service["spec"]["canaryTrafficPercent"] = canary_traffic_percent
        logger.info(f"Canary deployment: {canary_traffic_percent}% traffic to canary")

    # Create or update InferenceService
    custom_api = client.CustomObjectsApi()

    try:
        # Try to get existing service
        existing = custom_api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=service_name
        )

        # Update existing service
        logger.info(f"Updating existing InferenceService: {service_name}")
        custom_api.patch_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=service_name,
            body=inference_service
        )
        status = f"InferenceService '{service_name}' updated successfully"

    except client.exceptions.ApiException as e:
        if e.status == 404:
            # Create new service
            logger.info(f"Creating new InferenceService: {service_name}")
            custom_api.create_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                body=inference_service
            )
            status = f"InferenceService '{service_name}' created successfully"
        else:
            raise

    logger.info(status)
    logger.info(f"Model deployed from: {model.path}")

    return status
