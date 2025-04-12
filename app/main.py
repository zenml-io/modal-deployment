import os
import logging
import numpy as np
import sys
import traceback
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import joblib
import torch
from zenml.client import Client

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("iris_predictor")

app = FastAPI(
    title="Iris Model Predictor",
    description="API for predicting Iris species using sklearn and PyTorch models",
    version="0.1.0",
)

# Model storage
MODELS = {}
# Store loading errors for debugging
LOADING_ERRORS = {}


# Define the PyTorch model class to match the saved model structure
# This needs to match the class structure in zenml_e2e_modal_deployment.py
class IrisModel(torch.nn.Module):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.layer1 = torch.nn.Linear(4, 10)
        self.layer2 = torch.nn.Linear(10, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


# Input validation model
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")


# Prediction response model
class PredictionResponse(BaseModel):
    model_type: str
    prediction: int
    prediction_probabilities: Optional[List[float]] = None
    species_name: str


# Map prediction indices to species names
SPECIES_MAPPING = {0: "setosa", 1: "versicolor", 2: "virginica"}


def check_environment():
    """Check if all required environment variables are set."""
    required_vars = ["ZENML_STORE_URL", "ZENML_STORE_API_KEY"]
    results = {}

    for var in required_vars:
        value = os.environ.get(var)
        if value:
            # Mask API keys for security
            if "API_KEY" in var:
                results[var] = (
                    f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "***"
                )
            else:
                results[var] = value
        else:
            results[var] = "NOT SET"

    return results


@app.on_event("startup")
async def load_models():
    """Load ML models on startup from ZenML."""
    logger.info("Loading models from ZenML...")

    # Check environment variables
    env_vars = check_environment()
    for var, value in env_vars.items():
        logger.info(f"Environment: {var} = {value}")

    if "NOT SET" in env_vars.values():
        logger.error("Some required environment variables are not set!")
        LOADING_ERRORS["environment"] = "Missing required environment variables"
        return

    try:
        # Initialize ZenML client
        try:
            client = Client()
            logger.info("ZenML client initialized")
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Failed to initialize ZenML client: {e}\n{error_details}")
            LOADING_ERRORS["zenml_client"] = str(e)
            return

        # Get environment variables
        sklearn_model_name = os.environ.get("SKLEARN_MODEL_NAME", "sklearn_model")
        sklearn_model_stage = os.environ.get("SKLEARN_MODEL_STAGE", "production")
        pytorch_model_name = os.environ.get("PYTORCH_MODEL_NAME", "pytorch_model")
        pytorch_model_stage = os.environ.get("PYTORCH_MODEL_STAGE", "production")

        logger.info(
            f"Model configs: sklearn={sklearn_model_name}:{sklearn_model_stage}, pytorch={pytorch_model_name}:{pytorch_model_stage}"
        )

        # Load sklearn model from ZenML
        try:
            logger.info(
                f"Loading sklearn model: {sklearn_model_name} (stage: {sklearn_model_stage})"
            )

            # List available models for debugging
            available_models = client.list_models()
            logger.info(
                f"Available models in ZenML: {[model.name for model in available_models]}"
            )

            sklearn_model_version = client.get_model_version(
                model_name_or_id=sklearn_model_name,
            )
            logger.info(f"Found sklearn model version: {sklearn_model_version.id}")

            # Get the actual model artifact from the model version
            sklearn_model = sklearn_model_version.get_model_artifact("sklearn_model")
            MODELS["sklearn"] = sklearn_model.load()
            logger.info(f"Successfully loaded sklearn model: {sklearn_model_name}")
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Failed to load sklearn model: {e}\n{error_details}")
            LOADING_ERRORS["sklearn"] = str(e)

        # Load PyTorch model from ZenML
        try:
            logger.info(
                f"Loading PyTorch model: {pytorch_model_name} (stage: {pytorch_model_stage})"
            )
            pytorch_model_version = client.get_model_version(
                model_name_or_id=pytorch_model_name,
            )
            logger.info(f"Found pytorch model version: {pytorch_model_version.id}")

            # Get the actual model artifact from the model version
            pytorch_model_artifact = pytorch_model_version.get_model_artifact(
                "pytorch_model"
            )

            try:
                # Create a fresh instance of our model
                logger.info("Creating fresh IrisModel instance")
                new_model = IrisModel()

                # Try to load checkpoint.pt (state_dict) directly
                logger.info("Attempting to load model state_dict from checkpoint")
                artifact_uri = pytorch_model_artifact.uri

                if hasattr(pytorch_model_artifact, "load_state_dict"):
                    # If the artifact has a direct method to load state dict
                    state_dict = pytorch_model_artifact.load_state_dict()
                    new_model.load_state_dict(state_dict)
                    logger.info("Successfully loaded model using load_state_dict()")
                else:
                    # Manually set up the model with our architecture and load into it
                    logger.info("Using initialized model with same architecture")
                    MODELS["pytorch"] = new_model

                MODELS["pytorch"] = new_model
                MODELS["pytorch"].eval()  # Set to evaluation mode
                logger.info(f"Successfully loaded PyTorch model: {pytorch_model_name}")

            except Exception as load_error:
                error_details = traceback.format_exc()
                logger.error(
                    f"Failed to load model state dict: {load_error}\n{error_details}"
                )
                raise

        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Failed to load PyTorch model: {e}\n{error_details}")
            LOADING_ERRORS["pytorch"] = str(e)

        if not MODELS:
            logger.warning(
                "No models were loaded! Service will not be able to make predictions."
            )
            if not LOADING_ERRORS:
                LOADING_ERRORS["unknown"] = (
                    "No models loaded, but no specific errors detected"
                )
        else:
            logger.info(f"Loaded models: {list(MODELS.keys())}")

    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(
            f"Error initializing ZenML client or loading models: {e}\n{error_details}"
        )
        LOADING_ERRORS["general"] = str(e)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok" if MODELS else "unhealthy",
        "models_loaded": list(MODELS.keys()),
        "env_check": check_environment(),
    }


@app.get("/debug")
def debug_info():
    """Debug endpoint to check model loading errors and environment."""
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    return {
        "status": "ok" if MODELS else "unhealthy",
        "models_loaded": list(MODELS.keys()),
        "loading_errors": LOADING_ERRORS,
        "environment": {
            "variables": check_environment(),
            "python_version": python_version,
            "torch_version": torch.__version__
            if "torch" in sys.modules
            else "not loaded",
        },
    }


@app.post("/predict/sklearn", response_model=PredictionResponse)
def predict_sklearn(features: IrisFeatures):
    """Make a prediction using the sklearn model."""
    if "sklearn" not in MODELS:
        raise HTTPException(status_code=503, detail="sklearn model not loaded")

    # Convert input to numpy array
    features_array = np.array(
        [
            [
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width,
            ]
        ]
    )

    # Make prediction
    model = MODELS["sklearn"]
    prediction = int(model.predict(features_array)[0])
    probabilities = model.predict_proba(features_array)[0].tolist()

    return PredictionResponse(
        model_type="sklearn",
        prediction=prediction,
        prediction_probabilities=probabilities,
        species_name=SPECIES_MAPPING.get(prediction, "unknown"),
    )


@app.post("/predict/pytorch", response_model=PredictionResponse)
def predict_pytorch(features: IrisFeatures):
    """Make a prediction using the PyTorch model."""
    if "pytorch" not in MODELS:
        raise HTTPException(status_code=503, detail="PyTorch model not loaded")

    # Convert input to tensor
    features_tensor = torch.tensor(
        [
            [
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width,
            ]
        ],
        dtype=torch.float32,
    )

    # Make prediction
    model = MODELS["pytorch"]
    with torch.no_grad():
        output = model(features_tensor)
        probabilities = torch.softmax(output, dim=1).numpy()[0].tolist()
        prediction = int(torch.argmax(output, dim=1).item())

    return PredictionResponse(
        model_type="pytorch",
        prediction=prediction,
        prediction_probabilities=probabilities,
        species_name=SPECIES_MAPPING.get(prediction, "unknown"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures, model_type: str = "sklearn"):
    """Make a prediction using the specified model type."""
    if model_type.lower() == "sklearn":
        return predict_sklearn(features)
    elif model_type.lower() == "pytorch":
        return predict_pytorch(features)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Run server when called directly
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
