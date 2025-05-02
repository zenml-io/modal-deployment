#!/usr/bin/env python3
"""Unified Modal + FastAPI Iris predictor (sklearn or PyTorch)."""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Union

os.environ["MODAL_AUTOMOUNT"] = "false"

import modal
from fastapi import FastAPI, HTTPException
from modal.output import enable_output
from modal.runner import deploy_app

from app.schemas import (
    ApiInfo,
    IrisFeatures,
    IrisModel,
    PredictionResponse,
)

# --- Constants ---
MODEL_NAME = "iris-classifier"
MODAL_SECRET_NAME = "modal-deployment-credentials"
SPECIES_MAPPING = {0: "Iris setosa", 1: "Iris versicolor", 2: "Iris virginica"}

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iris-predictor")

# --- Environment variables ---
framework = os.environ.get("MODEL_FRAMEWORK", "sklearn")
stage = os.environ.get("MODEL_STAGE", "main")
volume_name = os.environ.get("MODAL_VOLUME_NAME", "iris-staging-models")
sklearn_model_path = os.environ.get("SKLEARN_MODEL_PATH", "sklearn_model.pkl")
pytorch_model_path = os.environ.get("PYTORCH_MODEL_PATH", "pytorch_model.pth")


def create_modal_app(
    framework: str,
    stage: str,
    volume_name: str,
    python_version: str = "3.12.9",
):
    """Create a Modal app for a given framework, stage, and volume name."""
    # Create base image
    base_image = (
        modal.Image.debian_slim(python_version=python_version)
        .pip_install(
            "numpy",
            "fastapi[standard]",
            "pydantic",
            "uvicorn",
            "modal",
            "scikit-learn" if framework == "sklearn" else "torch",
        )
        .add_local_python_source("app")
    )

    app_config = {
        "image": base_image,
        "secrets": [modal.Secret.from_name(MODAL_SECRET_NAME)],
    }

    if volume_name and volume_name.strip():
        try:
            volume = modal.Volume.from_name(volume_name)
            app_config["volumes"] = {"/app": volume}
            logger.info(f"Added volume {volume_name} to app")
        except Exception as e:
            logger.warning(f"Could not add volume {volume_name}: {e}")

    return modal.App(f"{framework}-iris-{stage}", **app_config)


# Create initial app instance
app = create_modal_app(framework, stage, volume_name)


# --- Define functions at the global scope without decorators ---


def _load_model() -> Any:
    """Load model into memory from Modal volume."""
    architecture_config = os.environ.get("MODEL_ARCHITECTURE", "{}")

    print("Volume root listing:", os.listdir("/app"))

    if framework == "sklearn" and sklearn_model_path:
        import pickle

        model_path = sklearn_model_path
        if not os.path.isabs(model_path):
            model_path = os.path.join("/app", model_path)

        if os.path.exists(model_path):
            logger.info(f"Loading sklearn model from volume: {model_path}")
            with open(model_path, "rb") as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    elif framework == "pytorch" and pytorch_model_path:
        # Only import torch when using the pytorch framework
        import torch

        # Load PyTorch model from volume
        model_path = pytorch_model_path
        if not os.path.isabs(model_path):
            model_path = os.path.join("/app", model_path)

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


def _predict(features: List[float]) -> Dict[str, Union[int, List[float], str]]:
    """Run inference using the pre-loaded model."""
    try:
        model = load_model.remote()
        if framework == "sklearn":
            import numpy as np

            # Convert input to numpy array
            input_array = np.array([features])

            prediction = int(model.predict(input_array)[0])
            probabilities = model.predict_proba(input_array)[0].tolist()
        else:
            # Only import torch when using the pytorch framework
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


def _create_fastapi_app() -> FastAPI:
    """Create FastAPI app for Iris prediction."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # warm the model when the container starts
        try:
            load_model.remote()
            logger.info("Model loaded on startup")
        except Exception as e:
            logger.error(f"Warm-up failed: {e}")
        yield

    web_app = FastAPI(
        title="Iris Prediction API",
        version="1.0.0",
        description=f"{MODEL_NAME} ({framework})",
        lifespan=lifespan,
    )

    @web_app.get("/", response_model=ApiInfo)
    async def root() -> Dict[str, Union[str, int]]:
        logger.info("Root endpoint called")
        actual_url = fastapi_app.web_url

        return {
            "message": f"({framework.capitalize()}) Iris Model Prediction API",
            "deployment_id": f"{framework}-iris-predictor-{stage}",
            "endpoints": {
                "root": actual_url,
                "health": f"{actual_url}/health",
                "predict": f"{actual_url}/predict",
                "url": f"{actual_url}/url",
            },
            "model": MODEL_NAME,
            "framework": framework,
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
        }

    @web_app.get("/url")
    async def get_url() -> Dict[str, str]:
        """Return the URL of this endpoint."""
        logger.info("URL endpoint called")
        return {
            "web_url": fastapi_app.web_url,
        }

    @web_app.get("/health")
    async def health() -> Dict[str, str]:
        """Health check endpoint."""
        logger.info("Health check called")
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
        }

    @web_app.post("/predict", response_model=PredictionResponse)
    async def predict_endpoint(features: IrisFeatures):
        """Prediction endpoint."""
        logger.info("Prediction request received")
        try:
            result = predict.remote(
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
        except Exception as e:
            logger.exception(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return web_app


# Create initial functions with the initial app
load_model = app.function()(_load_model)
predict = app.function()(_predict)
api_label = f"{framework}-iris-predictor-{stage}"
fastapi_app = app.function()(modal.asgi_app(label=api_label)(_create_fastapi_app))


@app.local_entrypoint()
def main(
    framework: str = "sklearn",
    stage: str = "staging",
    volume: str = "iris-staging-models",
    model_path: str = "sklearn_model.pkl",
    python_version: str = "3.12.9",
):
    """Main entrypoint for the application."""
    # Set environment variables
    os.environ.update(
        {
            "MODEL_FRAMEWORK": framework.lower(),
            "MODEL_STAGE": stage,
            "MODAL_VOLUME_NAME": volume or "",
        }
    )

    # Set model path if provided
    if model_path:
        if framework.lower() == "sklearn":
            os.environ["SKLEARN_MODEL_PATH"] = model_path
        else:
            os.environ["PYTORCH_MODEL_PATH"] = model_path

    # Create a fresh app instance
    global app, load_model, predict, fastapi_app
    app = create_modal_app(
        framework=framework,
        stage=stage,
        volume_name=volume,
        python_version=python_version,
    )

    # Rebind the global functions to the new app instance
    load_model = app.function()(_load_model)
    predict = app.function()(_predict)
    api_label = f"{framework}-iris-predictor-{stage}"
    fastapi_app = app.function()(modal.asgi_app(label=api_label)(_create_fastapi_app))

    # Deploy the app with all functions properly registered
    with enable_output():
        deploy_result = deploy_app(
            app,
            name=f"{framework}-iris-{stage}",
            environment_name=stage,
        )

    # Get the URL of the deployed app
    fastapi_url = None
    # Method 1: Try to get URL directly from the fastapi_app
    try:
        # Access the web_url attribute directly after deployment
        fastapi_url = fastapi_app.web_url
        logger.info(f"Got URL from fastapi_app.web_url: {fastapi_url}")
    except Exception as e:
        logger.warning(f"Could not get URL from fastapi_app directly: {e}")

        # Method 2: Construct URL from workspace name and label
        workspace_name = getattr(deploy_result, "workspace_name", "marwan-ext")
        fastapi_url = f"https://{workspace_name}-{stage}--{api_label}.modal.run"
        logger.info(f"Constructed URL with stage: {fastapi_url}")

    logger.info(f"Serving Iris Predictor on {api_label}")
    logger.info(f"Deploy App Result: {deploy_result}")
    logger.info(f"API URL: {fastapi_url}")

    return deploy_result, fastapi_url


def run_deployment_entrypoint(**kwargs) -> Any:
    """Wrapper to invoke local_entrypoint for external callers."""
    return main(**kwargs)
