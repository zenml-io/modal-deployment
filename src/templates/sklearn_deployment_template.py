#!/usr/bin/env python3
"""
Modal deployment script for sklearn Iris model.
This template is designed to be used without modification -
it reads all configuration from the model's metadata.
"""

from datetime import datetime
from typing import Dict, List, Union

import modal
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from zenml.client import Client
from zenml.models.v2.core.model_version import ModelVersionResponse
from src.constants import MODAL_SECRET_NAME

# Model configuration - will be obtained from ZenML
MODEL_NAME = "iris_classification"
MODEL_STAGE = "latest"  # Default to latest version, will be updated by CLI args

# Generate a deployment ID using model stage instead of random UUID
DEPLOYMENT_ID = f"sklearn-iris-{MODEL_STAGE}"


# Define request/response models
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")


class PredictionResponse(BaseModel):
    prediction: int
    prediction_probabilities: List[float]
    species_name: str


# Map prediction indices to species names
SPECIES_MAPPING = {0: "setosa", 1: "versicolor", 2: "virginica"}

# Create Modal app
app = modal.App(DEPLOYMENT_ID)


def find_sklearn_model_version() -> ModelVersionResponse:
    """Find the most recent model version that uses the sklearn framework."""
    client = Client()

    # Get all model versions for our model
    all_versions = client.list_model_versions(
        model_name_or_id=MODEL_NAME,
        hydrate=True,
    )

    if not all_versions:
        raise ValueError(f"No model versions found for {MODEL_NAME}")

    # Sort versions by created timestamp, newest first
    all_versions = sorted(all_versions, key=lambda x: x.created, reverse=True)

    # Find versions that use the sklearn framework
    sklearn_versions = []
    for version in all_versions:
        # Look for framework in metadata
        if hasattr(version, "metadata") and version.metadata:
            # Check if run_metadata exists and has a framework attribute
            if hasattr(version.metadata, "run_metadata") and hasattr(
                version.metadata.run_metadata, "framework"
            ):
                if version.metadata.run_metadata.framework == "sklearn":
                    sklearn_versions.append(version)

    print(f"Found {len(sklearn_versions)} sklearn model versions")

    # If we found sklearn versions, use the most recent one
    if sklearn_versions:
        latest_version = sklearn_versions[0]
        print(f"Using sklearn model version: {latest_version.number}")
        return latest_version

    # If no sklearn versions found, just return the latest version
    if all_versions:
        latest_version = all_versions[0]
        print(
            f"No sklearn versions found, using latest model version: {latest_version.number}"
        )
        return latest_version

    raise ValueError(f"No model versions found for {MODEL_NAME}")


def get_python_version_from_metadata() -> str:
    """Get the Python version from the model metadata, or use a default."""
    model_version = find_sklearn_model_version()
    default_python_version = "3.10"  # Fallback default

    if hasattr(model_version, "metadata") and model_version.metadata:
        # Try to get Python version from deployment metadata
        if hasattr(model_version.metadata, "deployment"):
            deployment_metadata = model_version.metadata.deployment
            # Check if python_version is specified
            if hasattr(deployment_metadata, "python_version"):
                version = deployment_metadata.python_version
                print(f"Found Python version in metadata: {version}")
                return version

    print(f"Using default Python version: {default_python_version}")
    return default_python_version


def get_model_dependencies() -> List[str]:
    """Get the sklearn model's dependencies from its metadata."""
    model_version = find_sklearn_model_version()

    if model_version is None:
        raise ValueError(f"No model version found for {MODEL_NAME}")

    # Default dependencies that should always be included
    default_deps = [
        "scikit-learn",
        "numpy",
        "zenml",
        "pydantic",
        "fastapi",
        "modal",
        "uvicorn",
    ]

    if hasattr(model_version, "metadata") and model_version.metadata:
        # Print metadata for debugging
        if hasattr(model_version, "number"):
            print(f"Model version {model_version.number} metadata available")

        # First check deployment section
        dependencies = []
        if hasattr(model_version.metadata, "deployment"):
            deployment_metadata = model_version.metadata.deployment
            print(f"Found deployment metadata")

            # Try to get any dependencies we can find
            if hasattr(deployment_metadata, "dependencies"):
                dependencies = deployment_metadata.dependencies
                print(f"Found {len(dependencies)} dependencies in deployment metadata")
            elif hasattr(deployment_metadata, "sklearn_dependencies"):
                dependencies = deployment_metadata.sklearn_dependencies
                print(f"Found {len(dependencies)} dependencies in sklearn_dependencies")
            elif hasattr(deployment_metadata, "core_dependencies"):
                dependencies = deployment_metadata.core_dependencies
                print(f"Found {len(dependencies)} dependencies in core_dependencies")

            # If we found dependencies, use them plus our defaults
            if dependencies:
                # Combine with default dependencies
                all_deps = list(set(dependencies + default_deps))
                print(f"Using {len(all_deps)} dependencies from metadata + defaults")
                return all_deps

    # If we couldn't find dependencies, just use the defaults
    print(f"Using {len(default_deps)} default dependencies")
    return default_deps


# Get dependencies and Python version from model metadata
dependencies = " ".join(get_model_dependencies())
python_version = get_python_version_from_metadata()

# Create the image with the specified Python version from metadata
image = (
    modal.Image.debian_slim(python_version=python_version)
    .pip_install("uv")
    .run_commands(f"uv pip install --system --compile-bytecode {dependencies}")
)


# Define a direct prediction function
@app.function(
    image=image,
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
    include_source=True,
)
def predict_sklearn(features: List[float]) -> Dict[str, Union[int, List[float], str]]:
    """Standalone prediction function that loads the model and makes predictions."""
    try:
        # Connect to ZenML and load the model
        client = Client()

        # Find the latest model version
        all_versions = client.list_model_versions(model_name_or_id=MODEL_NAME)
        if not all_versions:
            return {"error": f"No model versions found for {MODEL_NAME}"}

        # Sort by creation time (newest first)
        model_version = sorted(all_versions, key=lambda x: x.created, reverse=True)[0]
        model = model_version.get_model().load()

        # Convert input to numpy array
        input_array = np.array([features])

        # Make prediction
        prediction = int(model.predict(input_array)[0])
        probabilities = model.predict_proba(input_array)[0].tolist()

        return {
            "prediction": prediction,
            "prediction_probabilities": probabilities,
            "species_name": SPECIES_MAPPING.get(prediction, "unknown"),
        }
    except Exception as e:
        return {"error": str(e)}


# Define the FastAPI app
@app.function(
    image=image,
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
    include_source=True,
)
@modal.asgi_app(label=f"sklearn-iris-api-{MODEL_STAGE}")
def fastapi_app() -> FastAPI:
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("sklearn-model-api")

    # Create FastAPI app
    web_app = FastAPI(
        title="Sklearn Iris Model Predictor",
        description="API for predicting Iris species using sklearn model",
        version="1.0.0",
    )

    @web_app.get("/")
    async def root() -> Dict[str, Union[str, int]]:
        logger.info("Root endpoint called")
        return {
            "message": "Sklearn Iris Model Prediction API",
            "deployment_id": DEPLOYMENT_ID,
            "model": MODEL_NAME,
            "implementation": "sklearn",
            "timestamp": datetime.now().isoformat(),
        }

    @web_app.get("/health")
    async def health() -> Dict[str, str]:
        logger.info("Health check endpoint called")
        return {"status": "healthy"}

    @web_app.post("/predict", response_model=PredictionResponse)
    async def predict(features: IrisFeatures):
        logger.info("Prediction request received")
        # Call the standalone prediction function
        result = predict_sklearn.remote(
            [
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width,
            ]
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    logger.info("FastAPI app initialized")
    return web_app


# Deployment command
if __name__ == "__main__":
    import argparse

    # Allow overriding the model stage from command line
    parser = argparse.ArgumentParser(description="Deploy sklearn model from ZenML")
    parser.add_argument("--stage", default=MODEL_STAGE, help="Model stage to deploy")
    args = parser.parse_args()

    # Update the global MODEL_STAGE if specified
    if args.stage:
        MODEL_STAGE = args.stage
        # Update deployment ID to use the specified stage
        DEPLOYMENT_ID = f"sklearn-iris-{MODEL_STAGE}"

    # Get the model version to be deployed
    try:
        model_version = find_sklearn_model_version()
        version_str = model_version.number
    except Exception as e:
        version_str = "unknown"
        print(f"Error resolving model version: {e}")

    print(
        f"Deploying {MODEL_NAME} (sklearn implementation, version {version_str}) as app: {DEPLOYMENT_ID}"
    )
    print(f"Using dependencies: {dependencies}")
    modal.serve(fastapi_app)
    print(f"Deployment completed with ID: {DEPLOYMENT_ID}")
