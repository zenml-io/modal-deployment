#!/usr/bin/env python3
"""
Modal deployment script for PyTorch Iris model.
This template is designed to be used without modification -
it reads all configuration from the model's metadata.
"""

import argparse
from datetime import datetime
from typing import Any, Dict, List, Union

import modal
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from zenml.client import Client
from zenml.models.v2.core.model_version import ModelVersionResponse

from src.constants import (
    MODAL_SECRET_NAME,
    MODEL_NAME,
    SPECIES_MAPPING,
)


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


def find_pytorch_model_version() -> ModelVersionResponse:
    """Find the most recent model version that uses the pytorch framework."""
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

    # Find versions that use the PyTorch framework
    pytorch_versions = []
    for version in all_versions:
        # Look for framework in metadata
        if hasattr(version, "metadata") and version.metadata:
            # Check if run_metadata exists and has a framework attribute
            if hasattr(version.metadata, "run_metadata") and hasattr(
                version.metadata.run_metadata, "framework"
            ):
                if version.metadata.run_metadata.framework == "pytorch":
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
            elif hasattr(deployment_metadata, "pytorch_dependencies"):
                dependencies = deployment_metadata.pytorch_dependencies
                print(f"Found {len(dependencies)} dependencies in pytorch_dependencies")
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


def get_model_architecture_from_metadata() -> Dict[str, Any]:
    """Get the model architecture from metadata, if available."""
    model_version = find_pytorch_model_version()

    # Default architecture
    default_arch = {"input_dim": 4, "hidden_dim": 10, "output_dim": 3}

    if hasattr(model_version, "metadata") and model_version.metadata:
        # Check for architecture at top level
        if hasattr(model_version.metadata, "architecture"):
            print(f"Found architecture in top-level metadata")
            return model_version.metadata.architecture

        # Try to get architecture from deployment metadata
        if hasattr(model_version.metadata, "deployment"):
            deployment_metadata = model_version.metadata.deployment
            if hasattr(deployment_metadata, "architecture"):
                print(f"Found architecture in deployment metadata")
                return deployment_metadata.architecture

    print(f"Using default architecture: {default_arch}")
    return default_arch


# Define the PyTorch model class to match the saved model structure
class IrisModel(torch.nn.Module):
    """PyTorch neural network for Iris classification task."""

    def __init__(self, input_dim=4, hidden_dim=10, output_dim=3):
        super(IrisModel, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for IrisModel."""
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


def get_model_stage_from_args() -> str:
    """Get the deployment stage from the CLI args, or default to 'latest'."""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy PyTorch model from ZenML")
    parser.add_argument("--stage", default="latest", help="Model stage to deploy")
    args, _ = parser.parse_known_args()
    return args.stage


def get_deployment_id(model_stage: str) -> str:
    """Get deployment ID given the model stage."""
    return f"pytorch-iris-{model_stage}"


def predict_pytorch_factory(app, image):
    @app.function(
        image=image,
        secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
        include_source=True,
    )
    def predict_pytorch(
        features: List[float],
    ) -> Dict[str, Union[int, List[float], str]]:
        """Standalone prediction function that loads the model and makes predictions."""
        try:
            # Connect to ZenML and load the model
            client = Client()

            # Find the latest model version
            all_versions = client.list_model_versions(model_name_or_id=MODEL_NAME)
            if not all_versions:
                return {"error": f"No model versions found for {MODEL_NAME}"}

            # Sort by creation time (newest first)
            model_version = sorted(all_versions, key=lambda x: x.created, reverse=True)[
                0
            ]

            # Get architecture parameters from metadata
            architecture = {"input_dim": 4, "hidden_dim": 10, "output_dim": 3}
            if hasattr(model_version, "metadata") and model_version.metadata:
                if hasattr(model_version.metadata, "architecture"):
                    architecture = model_version.metadata.architecture
                elif hasattr(model_version.metadata, "deployment") and hasattr(
                    model_version.metadata.deployment, "architecture"
                ):
                    architecture = model_version.metadata.deployment.architecture

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

    return predict_pytorch


def fastapi_app_factory(app, image, deployment_id, model_stage):
    @app.function(
        image=image,
        secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
        include_source=True,
    )
    @modal.asgi_app(label=f"pytorch-iris-api-{model_stage}")
    def fastapi_app() -> FastAPI:
        import logging

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("pytorch-model-api")
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
                "deployment_id": deployment_id,
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

    return fastapi_app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy PyTorch model from ZenML")
    parser.add_argument("--stage", default="latest", help="Model stage to deploy")
    args = parser.parse_args()

    model_stage = args.stage
    deployment_id = f"pytorch-iris-{model_stage}"

    app = modal.App(deployment_id)
    dependencies = " ".join(get_model_dependencies())
    python_version = get_python_version_from_metadata()

    image = (
        modal.Image.debian_slim(python_version=python_version)
        .pip_install("uv")
        .run_commands(f"uv pip install --system --compile-bytecode {dependencies}")
    )

    # Register factory-generated functions (Modal needs them at runtime)
    predict_pytorch = predict_pytorch_factory(app, image)
    fastapi_app = fastapi_app_factory(app, image, deployment_id, model_stage)

    modal.serve(fastapi_app)
    print(f"Deployment completed with ID: {deployment_id}")
