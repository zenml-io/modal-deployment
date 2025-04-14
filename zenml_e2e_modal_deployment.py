import os
import logging
import numpy as np
import sys
import traceback
from typing import List, Dict, Any, Optional, Tuple, Annotated
import torch
import datetime
import uuid
from pathlib import Path
from zenml import step, pipeline, Model, log_metadata, get_step_context
from zenml.client import Client
from zenml.integrations.registry import integration_registry
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from rich import print

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("zenml_deployment")

# Define our models
sklearn_model = Model(
    name="sklearn_model",
    license="MIT",
    description="Iris classification model using RandomForestClassifier",
)

pytorch_model = Model(
    name="pytorch_model",
    license="MIT",
    description="Iris classification model using PyTorch neural network",
)

MODAL_SECRET_NAME = "zenml-internal-service-account"


# Define a simple neural network model
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


@step
def get_stack_dependencies() -> Annotated[List[str], "dependencies"]:
    """Get the dependencies required by the active ZenML stack.

    Returns:
        List of dependency strings required by the stack components
    """
    logger.info("Collecting dependencies from active ZenML stack...")
    client = Client()
    active_stack = client.active_stack

    # Collect all integration requirements
    all_dependencies = []

    # Get artifact store requirements
    artifact_store = active_stack.artifact_store
    artifact_store_flavor = artifact_store.flavor
    try:
        artifact_store_deps = integration_registry.select_integration_requirements(
            artifact_store_flavor
        )
        all_dependencies.extend(artifact_store_deps)
        logger.info(
            f"Added {len(artifact_store_deps)} dependencies from artifact store ({artifact_store_flavor})"
        )
    except KeyError:
        logger.info(
            f"Artifact store flavor '{artifact_store_flavor}' is not in the integration registry, skipping dependencies"
        )

    # Check for other stack components and get their dependencies
    for component_name in ["orchestrator", "artifact_store", "image_builder"]:
        if hasattr(active_stack, component_name):
            component = getattr(active_stack, component_name)
            if component:
                component_flavor = component.flavor
                try:
                    component_deps = (
                        integration_registry.select_integration_requirements(
                            component_flavor
                        )
                    )
                    all_dependencies.extend(component_deps)
                    logger.info(
                        f"Added {len(component_deps)} dependencies from {component_name} ({component_flavor})"
                    )
                except KeyError:
                    logger.info(
                        f"{component_name.capitalize()} flavor '{component_flavor}' is not in the integration registry, skipping dependencies"
                    )

    # Add core dependencies
    core_deps = ["zenml", "pydantic", "fastapi", "modal", "uvicorn"]
    all_dependencies.extend(core_deps)
    logger.info(f"Added {len(core_deps)} core dependencies")

    # Add model-specific dependencies
    model_deps = ["scikit-learn", "numpy", "torch"]
    all_dependencies.extend(model_deps)
    logger.info(f"Added {len(model_deps)} model-specific dependencies")

    # Make sure there are no duplicates
    unique_deps = list(set(all_dependencies))

    logger.info(f"Collected {len(unique_deps)} unique dependencies from active stack")
    return unique_deps


@step(model=sklearn_model)
def train_sklearn_model() -> Annotated[RandomForestClassifier, "sklearn_model"]:
    """Train and register a sklearn RandomForestClassifier model."""
    logger.info("Training sklearn model...")

    # Load dataset
    X, y = load_iris(return_X_y=True)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    logger.info(f"Sklearn model training accuracy: {train_accuracy:.4f}")
    logger.info(f"Sklearn model testing accuracy: {test_accuracy:.4f}")

    # Log metadata to the model
    log_metadata(
        metadata={
            "metrics": {
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy),
            },
            "parameters": {"n_estimators": 100, "random_state": 42},
            "signature": {
                "inputs": [{"name": "X", "dtype": "float64", "shape": [-1, 4]}],
                "outputs": [{"name": "y", "dtype": "int64", "shape": [-1]}],
            },
        },
        infer_model=True,
    )

    # Set this version to production
    current_model = get_step_context().model
    current_model.set_stage("production", force=True)
    logger.info(
        f"Set sklearn model version {current_model.version} to production stage"
    )

    return model


@step(model=pytorch_model)
def train_pytorch_model() -> Annotated[torch.nn.Module, "pytorch_model"]:
    """Train and register a PyTorch neural network model."""
    logger.info("Training PyTorch model...")

    # Load dataset
    X, y = load_iris(return_X_y=True)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    # Create model instance
    model = IrisModel()

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        _, train_predicted = torch.max(train_outputs.data, 1)
        train_accuracy = (train_predicted == y_train_tensor).sum().item() / len(
            y_train_tensor
        )

        test_outputs = model(X_test_tensor)
        _, test_predicted = torch.max(test_outputs.data, 1)
        test_accuracy = (test_predicted == y_test_tensor).sum().item() / len(
            y_test_tensor
        )

    logger.info(f"PyTorch model training accuracy: {train_accuracy:.4f}")
    logger.info(f"PyTorch model testing accuracy: {test_accuracy:.4f}")

    # Log metadata to the model
    log_metadata(
        metadata={
            "metrics": {
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy),
            },
            "parameters": {
                "learning_rate": 0.01,
                "epochs": num_epochs,
                "hidden_dim": 10,
            },
            "signature": {
                "inputs": [{"name": "X", "dtype": "float32", "shape": [-1, 4]}],
                "outputs": [{"name": "logits", "dtype": "float32", "shape": [-1, 3]}],
            },
        },
        infer_model=True,
    )

    # Set this version to production
    current_model = get_step_context().model
    current_model.set_stage("production", force=True)
    logger.info(
        f"Set PyTorch model version {current_model.version} to production stage"
    )

    return model


@step
def create_modal_deployment_script(
    stack_dependencies: Annotated[List[str], "dependencies"],
) -> Tuple[str, str]:
    """Create separate Modal deployment scripts for sklearn and PyTorch models.

    Args:
        stack_dependencies: Dependencies required by the active ZenML stack

    Returns:
        Tuple containing paths to the sklearn and PyTorch deployment scripts
    """
    logger.info("Creating Modal deployment scripts for sklearn and PyTorch models...")

    # Create unique deployment IDs
    sklearn_deployment_id = f"sklearn-iris-{uuid.uuid4().hex[:8]}"
    pytorch_deployment_id = f"pytorch-iris-{uuid.uuid4().hex[:8]}"

    # Create scripts directory if it doesn't exist
    scripts_dir = Path("modal_deployment_scripts")
    scripts_dir.mkdir(exist_ok=True)

    # Define script paths
    sklearn_script_path = scripts_dir / f"deploy_sklearn_{sklearn_deployment_id}.py"
    pytorch_script_path = scripts_dir / f"deploy_pytorch_{pytorch_deployment_id}.py"

    # Add model-specific dependencies to stack dependencies
    sklearn_dependencies = list(
        set(
            stack_dependencies
            + [
                "scikit-learn",
                "numpy",
            ]
        )
    )

    pytorch_dependencies = list(
        set(
            stack_dependencies
            + [
                "torch",
                "numpy",
            ]
        )
    )

    # Create sklearn deployment script
    sklearn_script_content = f"""# Generated Modal deployment script for sklearn Iris model
# Generated at: {datetime.datetime.now().isoformat()}
# Deployment ID: {sklearn_deployment_id}

import os
import json
import numpy as np
import modal
from typing import Dict, List, Union, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from zenml.client import Client

# Deployment configuration
DEPLOYMENT_ID = "{sklearn_deployment_id}"
MODEL_NAME = "sklearn_model"
MODEL_STAGE = "production"
MODAL_SECRET_NAME = "{MODAL_SECRET_NAME}"

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
SPECIES_MAPPING = {{0: "setosa", 1: "versicolor", 2: "virginica"}}

# Stack dependencies collected from ZenML
DEPENDENCIES = {sklearn_dependencies}

# Create Modal app
app = modal.App(DEPLOYMENT_ID)

# Prepare the image with all required dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install(DEPENDENCIES)

# Define the model deployment class
@app.cls(
    image=image,
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
    scaledown_window=300
)
class SklearnModelDeployer:
    def __init__(self):
        self.model = None
        
    @modal.enter()
    def load_model(self):
        # Connect to ZenML and load the model
        try:
            client = Client()
            model_version = client.get_model_version(
                model_name_or_id=MODEL_NAME,
                stage=MODEL_STAGE
            )
            self.model = model_version.get_model().load()
            print(f"Model {{MODEL_NAME}} loaded successfully")
        except Exception as e:
            print(f"Error loading model: {{e}}")
            self.model = None
        
    @modal.method()
    def predict(self, features):
        if self.model is None:
            return {{"error": "Model not loaded"}}
            
        try:
            # Convert input to numpy array
            input_array = np.array([features])
            
            # Make prediction
            prediction = int(self.model.predict(input_array)[0])
            probabilities = self.model.predict_proba(input_array)[0].tolist()
            
            return {{
                "prediction": prediction,
                "prediction_probabilities": probabilities,
                "species_name": SPECIES_MAPPING.get(prediction, "unknown")
            }}
        except Exception as e:
            return {{"error": str(e)}}

# Define the FastAPI app
@app.function(
    image=image,
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
)
@modal.asgi_app(label=f"sklearn-iris-api")
def fastapi_app():
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("sklearn-model-api")

    # Create FastAPI app
    app = FastAPI(
        title="Sklearn Iris Model Predictor",
        description="API for predicting Iris species using sklearn model",
        version="1.0.0"
    )

    # Create deployer instance
    model_deployer = SklearnModelDeployer()

    @app.get("/")
    async def root():
        logger.info("Root endpoint called")
        return {{
            "message": "Sklearn Iris Model Prediction API",
            "deployment_id": DEPLOYMENT_ID,
            "model": MODEL_NAME,
            "timestamp": datetime.now().isoformat()
        }}

    @app.get("/health")
    async def health():
        logger.info("Health check endpoint called")
        return {{"status": "healthy"}}

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(features: IrisFeatures):
        logger.info("Prediction request received")
        result = model_deployer.predict.remote([
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ])
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result

    logger.info("FastAPI app initialized")
    return app

# Deployment command
if __name__ == "__main__":
    print(f"Deploying app: {{DEPLOYMENT_ID}}")
    print(f"Using dependencies: {{DEPENDENCIES}}")
    modal.serve(fastapi_app)
    print(f"Deployment completed with ID: {{DEPLOYMENT_ID}}")
"""

    # Create PyTorch deployment script
    pytorch_script_content = f"""# Generated Modal deployment script for PyTorch Iris model
# Generated at: {datetime.datetime.now().isoformat()}
# Deployment ID: {pytorch_deployment_id}

import os
import json
import numpy as np
import modal
import torch
from typing import Dict, List, Union, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from zenml.client import Client

# Deployment configuration
DEPLOYMENT_ID = "{pytorch_deployment_id}"
MODEL_NAME = "pytorch_model"
MODEL_STAGE = "production"
MODAL_SECRET_NAME = "{MODAL_SECRET_NAME}"

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
SPECIES_MAPPING = {{0: "setosa", 1: "versicolor", 2: "virginica"}}

# Stack dependencies collected from ZenML
DEPENDENCIES = {pytorch_dependencies}

# Define the PyTorch model class to match the saved model structure
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

# Create Modal app
app = modal.App(DEPLOYMENT_ID)

# Prepare the image with all required dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install(DEPENDENCIES)

# Define the model deployment class
@app.cls(
    image=image,
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
    scaledown_window=300
)
class PyTorchModelDeployer:
    def __init__(self):
        self.model = None
        
    @modal.enter()
    def load_model(self):
        # Connect to ZenML and load the model
        try:
            client = Client()
            model_version = client.get_model_version(
                model_name_or_id=MODEL_NAME,
                stage=MODEL_STAGE
            )
            
            # Create a fresh instance of our model
            new_model = IrisModel()
            
            # Load the model
            model_artifact = model_version.get_model()
            if hasattr(model_artifact, "load_state_dict"):
                state_dict = model_artifact.load_state_dict()
                new_model.load_state_dict(state_dict)
            
            self.model = new_model
            self.model.eval()  # Set to evaluation mode
            print(f"Model {{MODEL_NAME}} loaded successfully")
        except Exception as e:
            print(f"Error loading model: {{e}}")
            self.model = None
        
    @modal.method()
    def predict(self, features):
        if self.model is None:
            return {{"error": "Model not loaded"}}
            
        try:
            # Convert input to tensor
            features_tensor = torch.tensor(
                [features],
                dtype=torch.float32,
            )
            
            # Make prediction
            with torch.no_grad():
                output = self.model(features_tensor)
                probabilities = torch.softmax(output, dim=1).numpy()[0].tolist()
                prediction = int(torch.argmax(output, dim=1).item())
            
            return {{
                "prediction": prediction,
                "prediction_probabilities": probabilities,
                "species_name": SPECIES_MAPPING.get(prediction, "unknown")
            }}
        except Exception as e:
            return {{"error": str(e)}}

# Define the FastAPI app
@app.function(
    image=image,
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
)
@modal.asgi_app(label=f"pytorch-iris-api")
def fastapi_app():
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("pytorch-model-api")

    # Create FastAPI app
    app = FastAPI(
        title="PyTorch Iris Model Predictor",
        description="API for predicting Iris species using PyTorch model",
        version="1.0.0"
    )

    # Create deployer instance
    model_deployer = PyTorchModelDeployer()

    @app.get("/")
    async def root():
        logger.info("Root endpoint called")
        return {{
            "message": "PyTorch Iris Model Prediction API",
            "deployment_id": DEPLOYMENT_ID,
            "model": MODEL_NAME,
            "timestamp": datetime.now().isoformat()
        }}

    @app.get("/health")
    async def health():
        logger.info("Health check endpoint called")
        return {{"status": "healthy"}}

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(features: IrisFeatures):
        logger.info("Prediction request received")
        result = model_deployer.predict.remote([
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ])
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result

    logger.info("FastAPI app initialized")
    return app

# Deployment command
if __name__ == "__main__":
    print(f"Deploying app: {{DEPLOYMENT_ID}}")
    print(f"Using dependencies: {{DEPENDENCIES}}")
    modal.serve(fastapi_app)
    print(f"Deployment completed with ID: {{DEPLOYMENT_ID}}")
"""

    # Write the scripts to disk
    with open(sklearn_script_path, "w") as f:
        f.write(sklearn_script_content)

    with open(pytorch_script_path, "w") as f:
        f.write(pytorch_script_content)

    logger.info(f"Created sklearn deployment script at {sklearn_script_path}")
    logger.info(f"Created PyTorch deployment script at {pytorch_script_path}")

    return (str(sklearn_script_path), str(pytorch_script_path))


@pipeline(enable_cache=False, name="iris_model_training_and_deployment")
def train_model_pipeline():
    """ZenML pipeline that trains, registers, and deploys Iris classification models.

    This pipeline:
    1. Trains a scikit-learn RandomForestClassifier
    2. Trains a PyTorch neural network
    3. Registers both models in the ZenML Model Registry
    4. Collects the active stack dependencies
    5. Creates separate Modal deployment scripts for each model with the required dependencies
    """
    sklearn_model = train_sklearn_model()
    pytorch_model = train_pytorch_model()
    stack_dependencies = get_stack_dependencies()
    create_modal_deployment_script(stack_dependencies=stack_dependencies)


if __name__ == "__main__":
    train_model_pipeline()
