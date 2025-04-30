#!/usr/bin/env python3
"""Unified Modal + FastAPI Iris predictor (sklearn or PyTorch).

Reads configuration from ZenML model metadata or YAML config.
"""

import argparse
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Union

import modal
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from zenml.client import Client
from zenml.models.v2.core.model_version import ModelVersionResponse

from src.schemas.iris_model import IrisModel
from src.utils.constants import MODAL_SECRET_NAME, MODEL_NAME, SPECIES_MAPPING
from src.utils.yaml_config import get_config_value

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iris-api")

# --- Default config values (used at import time) ---
framework = get_config_value("model.framework", "sklearn")
stage = os.getenv("MODEL_STAGE", "latest")

# --- Modal App ---
app = modal.App(f"{framework}-iris-{stage}")

# --- Environment variables ---
volume_name = os.environ.get("MODAL_VOLUME_NAME")
sklearn_model_path = os.environ.get("SKLEARN_MODEL_PATH")
pytorch_model_path = os.environ.get("PYTORCH_MODEL_PATH")


# --- Model version selection ---
def find_model_version(framework: str) -> ModelVersionResponse:
    """Return newest model version matching `framework`, else fallback to newest overall."""
    client = Client()
    versions = sorted(
        client.list_model_versions(model_name_or_id=MODEL_NAME, hydrate=True),
        key=lambda v: v.created,
        reverse=True,
    )
    if not versions:
        raise RuntimeError(f"No versions found for {MODEL_NAME}")
    matched = [
        v
        for v in versions
        if getattr(getattr(v.metadata, "run_metadata", None), "framework", None)
        == framework
    ]
    latest_version = matched[0] if matched else versions[0]
    logger.info(f"Using {framework} model version: {latest_version.number}")
    return latest_version


# --- Configuration extraction ---
def get_deployment_config(model_version: ModelVersionResponse) -> Dict[str, Any]:
    """Extract deployment configuration from model metadata.

    Args:
        model_version: The model version to extract config from

    Returns:
        Dictionary with deployment configuration values
    """
    deploy_meta = getattr(model_version.metadata, "deployment", {}) or {}
    return {
        "python_version": getattr(deploy_meta, "python_version", "3.10"),
        "architecture": getattr(deploy_meta, "architecture", {}),
    }


def get_model_dependencies(
    model_version: ModelVersionResponse, framework: str
) -> List[str]:
    """Get dependencies from model metadata.

    Args:
        model_version: The model version to extract dependencies from
        framework: The ML framework ("sklearn" or "pytorch")

    Returns:
        List of package dependencies
    """
    deploy_meta = getattr(model_version.metadata, "deployment", {}) or {}
    base_deps = [
        "numpy",
        "zenml",
        "fastapi",
        "pydantic",
        "uvicorn",
        "modal",
        "scikit-learn" if framework == "sklearn" else "torch",
    ]

    # Get all dependencies, with framework-specific ones prioritized
    all_deps = list(
        {
            *getattr(deploy_meta, f"{framework}_dependencies", []),
            *getattr(deploy_meta, "dependencies", []),
            *base_deps,
        }
    )

    return all_deps


# Load model version
mv = find_model_version(framework)

# Get deployment configuration
config = get_deployment_config(mv)
dependencies = get_model_dependencies(mv, framework)
python_version = config["python_version"]
architecture = config["architecture"]

# --- Build Modal Image ---
image = modal.Image.debian_slim(python_version=python_version).pip_install(
    *dependencies
)


# --- Pydantic schemas ---
class IrisFeatures(BaseModel):
    """Request model for iris prediction."""

    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)


class PredictionResponse(BaseModel):
    """Response model for iris prediction."""

    prediction: int
    prediction_probabilities: List[float]
    species_name: str


# --- Model loader ---
@modal.function(
    image=image,
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
    shared=True,
    volumes={volume_name: "/models"} if volume_name else {},
)
def load_model() -> Any:
    """Load model into memory (called once per Modal worker)."""
    if framework == "sklearn" and sklearn_model_path:
        # Load sklearn model from volume
        import pickle

        if os.path.exists(sklearn_model_path):
            logger.info(f"Loading sklearn model from volume: {sklearn_model_path}")
            with open(sklearn_model_path, "rb") as f:
                return pickle.load(f)

        elif framework == "pytorch" and pytorch_model_path:
            # Load PyTorch model from volume

            if os.path.exists(pytorch_model_path):
                logger.info(
                    f"Loading PyTorch model from volume: {pytorch_model_path}"
                )
                model = IrisModel(**architecture)
                model.load_state_dict(torch.load(pytorch_model_path))
                model.eval()
                return model

        # Fallback to ZenML loading if env vars not available or paths don't exist
        logger.warning(
            "Modal volume paths not found, falling back to ZenML artifact loading"
        )
        mv_local = find_model_version(framework)
        art = mv_local.get_model()
        if framework == "sklearn":
            return art.load()
        # PyTorch branch
        model = IrisModel(**architecture)
        state = art.load_state_dict()
        model.load_state_dict(state)
        model.eval()
        return model


# --- Prediction function ---
@modal.function(
    image=image,
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
    shared=False,
)
def predict(features: List[float]) -> Dict[str, Union[int, List[float], str]]:
    """Run inference using the pre-loaded model."""
    try:
        model = load_model.call()
        if framework == "sklearn":
            import numpy as np

            # Convert input to numpy array
            input_array = np.array([features])

            prediction = int(model.predict(input_array)[0])
            probabilities = model.predict_proba(input_array)[0].tolist()
        else:
            import torch

            tensor = torch.tensor([features], dtype=torch.float32)
            out = model(tensor)
            probabilities = torch.softmax(out, dim=1).cpu().numpy()[0].tolist()
            prediction = int(torch.argmax(out, dim=1).item())

        return {
            "prediction": prediction,
            "prediction_probabilities": probabilities,
            "species_name": SPECIES_MAPPING.get(prediction, "unknown"),
        }
    except Exception as e:
        return {"error": str(e)}


# --- FastAPI application ---
@modal.asgi_app(label=f"iris-api-{stage}")
def fastapi_app() -> FastAPI:
    """Create FastAPI app for Iris prediction."""
    web_app = FastAPI(
        title="Iris Prediction API",
        version="1.0.0",
        description=f"{MODEL_NAME} ({framework})",
    )

    @web_app.get("/")
    async def root() -> Dict[str, Union[str, int]]:
        logger.info("Root endpoint called")
        return {
            "message": f"({framework.capitalize()}) Iris Model Prediction API",
            "deployment_id": f"{framework}-iris-{stage}",
            "model": MODEL_NAME,
            "framework": framework,
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
        }

    @web_app.get("/health")
    async def health() -> Dict[str, str]:
        """Health check endpoint."""
        logger.info("Health check endpoint called")
        return {"status": "healthy"}

    @web_app.post("/predict", response_model=PredictionResponse)
    async def predict(features: IrisFeatures):
        """Prediction endpoint."""
        logger.info("Prediction request received")
        result = predict.call(
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

    return web_app


# --- Startup hook: warm model ---
@fastapi_app.enter()
def startup() -> None:
    """Warm up the model on startup."""
    load_model.call()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy Iris model via Modal+FastAPI"
    )
    parser.add_argument(
        "--framework",
        default=get_config_value("model.framework", default="sklearn"),
        help="ML framework: sklearn or pytorch",
    )
    parser.add_argument(
        "--stage",
        default=os.getenv("MODEL_STAGE", "latest"),
        help="Deployment stage",
    )
    args = parser.parse_args()
    framework = args.framework.lower()
    stage = args.stage

    # Create a fresh app instance with CLI args
    app = modal.App(f"{framework}-iris-{stage}")

    modal.serve(fastapi_app)
    logger.info(f"Serving Iris API on {framework}-iris-{stage}")
