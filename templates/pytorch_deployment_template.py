#!/usr/bin/env python3
"""
Modal deployment script for PyTorch Iris model.
This template is designed to be used without modification -
it reads all configuration from the model's metadata.
"""

import os
import json
import uuid
import sys
import numpy as np
import modal
import torch
from typing import Dict, List, Union, Optional, Any
from zenml.models.v2.core.model_version import ModelVersionResponse
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from zenml.client import Client

# Generate a unique deployment ID
DEPLOYMENT_ID = f"pytorch-iris-{uuid.uuid4().hex[:8]}"

# Model configuration - will be obtained from ZenML
MODEL_NAME = "iris_classification"
MODEL_STAGE = "latest"  # Default to latest version

# Secret name in Modal
MODAL_SECRET_NAME = "zenml-internal-service-account"


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


def find_pytorch_model_version() -> ModelVersionResponse:
    """Find the most recent model version that uses the pytorch framework."""
    client = Client()

    # Get all model versions for our model
    all_versions = client.list_model_versions(model_name_or_id=MODEL_NAME)

    if not all_versions:
        raise ValueError(f"No model versions found for {MODEL_NAME}")

    # Sort versions by created timestamp, newest first
    all_versions = sorted(all_versions, key=lambda x: x.created, reverse=True)

    # Find versions that use the PyTorch framework
    pytorch_versions = []
    for version in all_versions:
        # Look for framework in metadata
        if hasattr(version, "metadata") and version.metadata:
            if version.metadata.get("framework") == "pytorch":
                pytorch_versions.append(version)

    print(f"Found {len(pytorch_versions)} PyTorch model versions")

    # If we found PyTorch versions, use the most recent one
    if pytorch_versions:
        latest_version = pytorch_versions[0]
        print(f"Using PyTorch model version: {latest_version.number}")
        return latest_version

    # If no PyTorch versions found, just return the latest version
    if all_versions:
        latest_version = all_versions[0]
        print(
            f"No PyTorch versions found, using latest model version: {latest_version.number}"
        )
        return latest_version

    raise ValueError(f"No model versions found for {MODEL_NAME}")


def get_python_version_from_metadata() -> str:
    """Get the Python version from the model metadata, or use a default."""
    model_version = find_pytorch_model_version()
    default_python_version = "3.10"  # Fallback default

    if hasattr(model_version, "metadata") and model_version.metadata:
        # Try to get Python version from deployment metadata
        deployment_metadata = model_version.metadata.get("deployment", {})

        # Check if python_version is specified
        if "python_version" in deployment_metadata:
            version = deployment_metadata["python_version"]
            print(f"Found Python version in metadata: {version}")
            return version

    print(f"Using default Python version: {default_python_version}")
    return default_python_version


def get_model_dependencies() -> List[str]:
    """Get the PyTorch model's dependencies from its metadata."""
    model_version = find_pytorch_model_version()

    if model_version is None:
        raise ValueError(f"No model version found for {MODEL_NAME}")

    # Default dependencies that should always be included
    default_deps = [
        "torch",
        "numpy",
        "zenml",
        "pydantic",
        "fastapi",
        "modal",
        "uvicorn",
    ]

    if hasattr(model_version, "metadata") and model_version.metadata:
        # Print metadata for debugging
        print(
            f"Model version {model_version.number} metadata keys: {list(model_version.metadata.keys())}"
        )

        # First check deployment section
        deployment_metadata = model_version.metadata.get("deployment", {})
        print(f"Deployment metadata keys: {list(deployment_metadata.keys())}")

        # Try to get any dependencies we can find
        dependencies = []

        # Check all possible places where dependencies might be stored
        if "dependencies" in deployment_metadata:
            dependencies = deployment_metadata["dependencies"]
            print(f"Found {len(dependencies)} dependencies in deployment metadata")
        elif "pytorch_dependencies" in deployment_metadata:
            dependencies = deployment_metadata["pytorch_dependencies"]
            print(f"Found {len(dependencies)} dependencies in pytorch_dependencies")
        elif "core_dependencies" in deployment_metadata:
            dependencies = deployment_metadata["core_dependencies"]
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


def get_model_architecture_from_metadata() -> Dict[str, Any]:
    """Get the model architecture from metadata, if available."""
    model_version = find_pytorch_model_version()

    # Default architecture
    default_arch = {"input_dim": 4, "hidden_dim": 10, "output_dim": 3}

    if hasattr(model_version, "metadata") and model_version.metadata:
        # Print architecture related info for debugging
        metadata = model_version.metadata
        if "architecture" in metadata:
            print(
                f"Found architecture in top-level metadata: {metadata['architecture']}"
            )
            return metadata["architecture"]

        # Try to get architecture from deployment metadata
        deployment_metadata = metadata.get("deployment", {})
        if "architecture" in deployment_metadata:
            print(
                f"Found architecture in deployment metadata: {deployment_metadata['architecture']}"
            )
            return deployment_metadata["architecture"]

    print(f"Using default architecture: {default_arch}")
    return default_arch


# Define the PyTorch model class to match the saved model structure
class IrisModel(torch.nn.Module):
    def __init__(self, input_dim=4, hidden_dim=10, output_dim=3):
        super(IrisModel, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


# Create Modal app
app = modal.App(DEPLOYMENT_ID)

# Get dependencies and Python version from model metadata and prepare the image
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
def predict_pytorch(features: List[float]) -> Dict[str, Union[int, List[float], str]]:
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

        # Get architecture parameters from metadata
        architecture = {"input_dim": 4, "hidden_dim": 10, "output_dim": 3}
        if hasattr(model_version, "metadata") and model_version.metadata:
            metadata = model_version.metadata
            if "architecture" in metadata:
                architecture = metadata["architecture"]
            elif "deployment" in metadata and "architecture" in metadata["deployment"]:
                architecture = metadata["deployment"]["architecture"]

        # Create a fresh instance of our model
        model = IrisModel(
            input_dim=architecture.get("input_dim", 4),
            hidden_dim=architecture.get("hidden_dim", 10),
            output_dim=architecture.get("output_dim", 3),
        )

        # Load the model weights
        model_artifact = model_version.get_model()
        if hasattr(model_artifact, "load_state_dict"):
            state_dict = model_artifact.load_state_dict()
            model.load_state_dict(state_dict)

        model.eval()  # Set to evaluation mode

        # Convert input to tensor
        features_tensor = torch.tensor(
            [features],
            dtype=torch.float32,
        )

        # Make prediction
        with torch.no_grad():
            output = model(features_tensor)
            probabilities = torch.softmax(output, dim=1).numpy()[0].tolist()
            prediction = int(torch.argmax(output, dim=1).item())

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
@modal.asgi_app(label=f"pytorch-iris-api")
def fastapi_app() -> FastAPI:
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("pytorch-model-api")

    # Create FastAPI app
    web_app = FastAPI(
        title="PyTorch Iris Model Predictor",
        description="API for predicting Iris species using PyTorch model",
        version="1.0.0",
    )

    @web_app.get("/")
    async def root() -> Dict[str, Union[str, int]]:
        logger.info("Root endpoint called")
        return {
            "message": "PyTorch Iris Model Prediction API",
            "deployment_id": DEPLOYMENT_ID,
            "model": MODEL_NAME,
            "implementation": "pytorch",
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
        result = predict_pytorch.remote(
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
    parser = argparse.ArgumentParser(description="Deploy PyTorch model from ZenML")
    parser.add_argument("--stage", default=MODEL_STAGE, help="Model stage to deploy")
    args = parser.parse_args()

    # Update the global MODEL_STAGE if specified
    if args.stage:
        MODEL_STAGE = args.stage

    # Get the model version to be deployed
    try:
        model_version = find_pytorch_model_version()
        version_str = model_version.number
    except Exception as e:
        version_str = "unknown"
        print(f"Error resolving model version: {e}")

    print(
        f"Deploying {MODEL_NAME} (PyTorch implementation, version {version_str}) as app: {DEPLOYMENT_ID}"
    )
    print(f"Using dependencies: {dependencies}")
    modal.serve(fastapi_app)
    print(f"Deployment completed with ID: {DEPLOYMENT_ID}")
