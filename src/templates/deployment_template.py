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

from src.schemas import (
    IrisFeatures,
    IrisModel,
    PredictionResponse,
)
from src.utils.constants import (
    MODAL_SECRET_NAME,
    MODEL_NAME,
    SPECIES_MAPPING,
)
from src.utils.model_utils import get_python_version_from_metadata

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iris-api")


# --- Environment variables ---
framework = os.environ.get("MODEL_FRAMEWORK", "sklearn")
stage = os.environ.get("MODEL_STAGE", "latest")
volume_name = os.environ.get("MODAL_VOLUME_NAME")
sklearn_model_path = os.environ.get("SKLEARN_MODEL_PATH")
pytorch_model_path = os.environ.get("PYTORCH_MODEL_PATH")
architecture_config = os.environ.get("MODEL_ARCHITECTURE", "{}")


def create_modal_app(
    framework: str,
    stage: str,
    volume_name: str,
):
    """Create a Modal app for a given framework, stage, and volume name."""
    volume = modal.Volume.from_name(volume_name) if volume_name else None
    python_version = get_python_version_from_metadata(framework)
    return modal.App(
        f"{framework}-iris-{stage}",
        mounts={"/models": volume} if volume else {},
        secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
        image=modal.Image.debian_slim(python_version=python_version).pip_install(
            "numpy",
            "fastapi",
            "pydantic",
            "uvicorn",
            "modal",
            "scikit-learn" if framework == "sklearn" else "torch",
        ),
    )


# Create app using environment variables
app = create_modal_app(framework, stage, volume_name)


# --- Model loader ---
@app.function(shared=True)
def load_model() -> Any:
    """Load model into memory from Modal volume (called once per Modal worker)."""
    if framework == "sklearn" and sklearn_model_path:
        import pickle

        model_path = sklearn_model_path
        if not os.path.isabs(model_path):
            model_path = os.path.join("/models", model_path)

        if os.path.exists(model_path):
            logger.info(f"Loading sklearn model from volume: {model_path}")
            with open(model_path, "rb") as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    elif framework == "pytorch" and pytorch_model_path:
        # Load PyTorch model from volume
        model_path = pytorch_model_path
        if not os.path.isabs(model_path):
            model_path = os.path.join("/models", model_path)

        if os.path.exists(model_path):
            logger.info(f"Loading PyTorch model from volume: {model_path}")
            model = IrisModel(**architecture_config)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            return model
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    else:
        raise ValueError(
            f"No model path provided for framework {framework}. "
            f"Set SKLEARN_MODEL_PATH or PYTORCH_MODEL_PATH environment variables."
        )


# --- Prediction function ---
@app.function(shared=False)
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
        logger.exception("Prediction error")
        return {"error": str(e)}


# --- FastAPI application ---
@app.asgi_app(label=f"iris-api-{stage}")
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
    async def predict_endpoint(features: IrisFeatures):
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
    try:
        load_model.call()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        # We don't raise here to allow the app to start,
        # but first prediction will fail if model can't be loaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy Iris model via Modal+FastAPI"
    )
    parser.add_argument(
        "--framework",
        default=os.environ.get("MODEL_FRAMEWORK", "sklearn"),
        help="ML framework: sklearn or pytorch",
    )
    parser.add_argument(
        "--stage",
        default=os.getenv("MODEL_STAGE", "latest"),
        help="Deployment stage",
    )
    parser.add_argument(
        "--volume",
        default=os.environ.get("MODAL_VOLUME_NAME"),
        help="Modal volume name containing the model",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to model file within the volume",
    )
    args = parser.parse_args()

    # Override with CLI args
    framework = args.framework.lower()
    stage = args.stage
    volume_name = args.volume

    if args.model_path:
        if framework == "sklearn":
            os.environ["SKLEARN_MODEL_PATH"] = args.model_path
        else:
            os.environ["PYTORCH_MODEL_PATH"] = args.model_path

    # Recreate app with CLI args
    app = create_modal_app(framework, stage, volume_name)

    modal.serve(fastapi_app)
    logger.info(f"Serving Iris API on {framework}-iris-{stage}")
