# model_deploy.py
from typing import Dict, List, Union, Optional
import numpy as np
import modal
from pydantic import BaseModel
import toml
from zenml.client import Client

from zenml.integrations.registry import integration_registry


class PredictionRequest(BaseModel):
    features: List[List[float]]


class PredictionResponse(BaseModel):
    predictions: List[Union[int, float]]
    probabilities: Optional[List[List[float]]] = None


app = modal.App("model-deployment")

with open("pyproject.toml", "r") as f:
    pyproject = toml.load(f)

# Extract base package names from pyproject.toml
pyproject_packages = [
    dep.split(">=")[0] for dep in pyproject["project"]["dependencies"]
]

# Collect integration requirements
integration_packages = []
for integration in ["aws", "s3", "sklearn", "numpy"]:
    requirements = integration_registry.select_integration_requirements(integration)
    integration_packages.extend(requirements)

# Combine all packages into a single list
all_packages = pyproject_packages + integration_packages

# Filter out any empty strings
all_packages = [pkg for pkg in all_packages if pkg.strip()]

# Properly quote packages that contain special characters like >, <, !=
quoted_packages = []
for pkg in all_packages:
    if any(char in pkg for char in [">", "<", "!", "~"]):
        quoted_packages.append(f'"{pkg}"')
    else:
        quoted_packages.append(pkg)

# Create a space-separated string of packages
package_list = " ".join(quoted_packages)

# Add explicit numpy constraint
if '"numpy' not in package_list:
    package_list += ' "numpy<2.0.0"'

image = (
    modal.Image.debian_slim(python_version="3.12.3")
    .add_local_file("pyproject.toml", remote_path="/root/pyproject.toml", copy=True)
    .pip_install("uv")
    .run_commands(f"uv pip install --system --compile-bytecode {package_list}")
)


# Define a class to encapsulate model loading and prediction
@app.cls(
    image=image, secrets=[modal.Secret.from_name("zenml-internal-service-account")]
)
class ModelDeployer:
    def __init__(self):
        self.sklearn_model = None

    @modal.enter()
    def load_models(self):
        # Load scikit-learn model
        try:
            zenml_client = Client()
            self.sklearn_model = zenml_client.get_artifact_version(
                name_id_or_prefix="sklearn_model"
            ).load()
            print("Scikit-learn model loaded successfully")
        except Exception as e:
            print(f"Error loading scikit-learn model: {e}")
            self.sklearn_model = None

    # Scikit-learn model prediction endpoint
    @modal.method()
    def predict_sklearn(self, input_data: List[List[float]]) -> Dict:
        if self.sklearn_model is None:
            return {"error": "Scikit-learn model not loaded"}

        # Convert input to numpy array
        input_array = np.array(input_data)

        # Make prediction
        predictions = self.sklearn_model.predict(input_array)

        # Get probabilities if model supports it
        probabilities = None
        if hasattr(self.sklearn_model, "predict_proba"):
            probabilities = self.sklearn_model.predict_proba(input_array).tolist()

        return {"predictions": predictions.tolist(), "probabilities": probabilities}


@app.function(
    image=image, secrets=[modal.Secret.from_name("zenml-internal-service-account")]
)
@modal.asgi_app(label="model-api")
def fastapi_app():
    from fastapi import FastAPI
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("model-api")

    # Create deployer instance
    model_deployer = ModelDeployer()

    # Create FastAPI app
    app = FastAPI(title="Model Deployment API")

    @app.get("/")
    async def root():
        logger.info("Root endpoint called")
        return {"message": "Welcome to the Model Deployment API"}

    @app.get("/health")
    async def health():
        logger.info("Health check endpoint called")
        return {"status": "healthy"}

    @app.post("/predict/sklearn")
    async def predict(request: PredictionRequest):
        # print out the current numpy version
        print(np.__version__)
        logger.info(f"Prediction request received with {len(request.features)} samples")
        result = model_deployer.predict_sklearn.remote(request.features)
        return result

    logger.info("FastAPI app initialized")
    return app


# For running the development server
if __name__ == "__main__":
    modal.serve(fastapi_app)
