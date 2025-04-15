import argparse
import os
import logging
import traceback
import importlib.util
import sys
import platform
from typing import List, Dict, Any, Tuple, Annotated, Optional
import torch
import datetime
import uuid
import shutil
import tempfile
from pathlib import Path
from zenml import step, pipeline, Model, log_metadata, get_step_context
from zenml.client import Client
from zenml.integrations.registry import integration_registry
from zenml.enums import ModelStages
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from rich import print

try:
    from modal.runner import deploy_app
    from modal.output import enable_output

    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False
    logging.warning(
        "Modal package not found. Deployment functionality will be limited."
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("zenml_deployment")

# Define a single model for both implementations
iris_model = Model(
    name="iris_classification",
    license="MIT",
    description="Iris classification model with multiple implementations (sklearn and PyTorch)",
)

MODAL_SECRET_NAME = "modal-deployment-credentials"


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
    """Get the dependencies required by the active ZenML stack and log them to model.

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

    # Add all model-specific dependencies
    model_deps = ["scikit-learn", "numpy", "torch"]
    all_dependencies.extend(model_deps)
    logger.info(f"Added {len(model_deps)} model-specific dependencies")

    # Make sure there are no duplicates
    unique_deps = list(set(all_dependencies))

    logger.info(f"Collected {len(unique_deps)} unique dependencies from active stack")

    # Log dependencies to the model if it exists
    try:
        # Look for existing versions of our model
        model_versions = client.list_model_versions(
            model_name_or_id="iris_classification"
        )

        if model_versions:
            # Get the latest version
            latest_version = sorted(
                model_versions, key=lambda x: x.created, reverse=True
            )[0]

            # Create deployment metadata for both implementations
            deployment_metadata = {
                "deployment": {
                    "core_dependencies": unique_deps,
                    "sklearn_dependencies": unique_deps + ["scikit-learn", "numpy"],
                    "pytorch_dependencies": unique_deps + ["torch", "numpy"],
                    "updated_at": datetime.datetime.now().isoformat(),
                    "modal_secret": MODAL_SECRET_NAME,
                }
            }

            # Log the updated metadata using the zenml log_metadata function
            log_metadata(
                metadata=deployment_metadata,
                model_name="iris_classification",
                model_version=latest_version.number,
            )
            logger.info(
                f"Logged dependencies to iris_classification model version {latest_version.number}"
            )
    except Exception as e:
        logger.warning(f"Could not log dependencies to existing model: {e}")
        logger.info("Dependencies will be logged during model training steps")

    return unique_deps


@step(model=iris_model)
def train_sklearn_model(
    stack_dependencies: Annotated[List[str], "dependencies"],
) -> Annotated[RandomForestClassifier, "sklearn_model"]:
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

    # Include deployment metadata directly in the model metadata
    sklearn_deps = stack_dependencies + ["scikit-learn", "numpy"]

    # Get the local Python version
    python_version = platform.python_version().rsplit(".", 1)[
        0
    ]  # Get major.minor version (e.g., "3.10")
    logger.info(f"Using local Python version: {python_version}")

    # Log metadata to the model
    log_metadata(
        metadata={
            "framework": "sklearn",
            "implementation": "RandomForestClassifier",
            "metrics": {
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy),
            },
            "parameters": {"n_estimators": 100, "random_state": 42},
            "signature": {
                "inputs": [{"name": "X", "dtype": "float64", "shape": [-1, 4]}],
                "outputs": [{"name": "y", "dtype": "int64", "shape": [-1]}],
            },
            # Add deployment metadata
            "deployment": {
                "framework": "sklearn",
                "dependencies": sklearn_deps,
                "created_at": datetime.datetime.now().isoformat(),
                "modal_secret": MODAL_SECRET_NAME,
                "python_version": python_version,
            },
        },
        infer_model=True,
    )

    # Get the current model - it will already be in "latest" stage by default
    current_model = get_step_context().model
    logger.info(
        f"Registered iris_classification sklearn model as version {current_model.version}"
    )

    return model


@step(model=iris_model)
def train_pytorch_model(
    stack_dependencies: Annotated[List[str], "dependencies"],
) -> Annotated[torch.nn.Module, "pytorch_model"]:
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

    # Get model architecture parameters to save in metadata
    architecture = {"input_dim": 4, "hidden_dim": 10, "output_dim": 3}

    # Include deployment metadata directly in the model metadata
    pytorch_deps = stack_dependencies + ["torch", "numpy"]

    # Get the local Python version
    python_version = platform.python_version().rsplit(".", 1)[
        0
    ]  # Get major.minor version (e.g., "3.10")
    logger.info(f"Using local Python version: {python_version}")

    # Log metadata to the model
    log_metadata(
        metadata={
            "framework": "pytorch",
            "implementation": "IrisModel",
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
            "architecture": architecture,
            # Add deployment metadata
            "deployment": {
                "framework": "pytorch",
                "dependencies": pytorch_deps,
                "created_at": datetime.datetime.now().isoformat(),
                "modal_secret": MODAL_SECRET_NAME,
                "architecture": architecture,
                "python_version": python_version,
            },
        },
        infer_model=True,
    )

    # Get the current model - it will already be in "latest" stage by default
    current_model = get_step_context().model
    logger.info(
        f"Registered iris_classification pytorch model as version {current_model.version}"
    )

    return model


def load_python_module(file_path):
    """Dynamically load a Python module from a file path."""
    module_name = Path(file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@step
def modal_deployment(
    deploy: bool = False,
    stream_logs: bool = False,
    app_prefix: str = "iris-model",
    promote_to_stage: Optional[str] = None,
) -> Tuple[str, str, Dict[str, Dict[str, Any]]]:
    """Create Modal deployment scripts using templates and optionally deploy them.

    Args:
        deploy: Whether to actually deploy the scripts using Modal
        stream_logs: Whether to stream logs from the deployments
        app_prefix: Prefix to use for app names
        promote_to_stage: If specified, promote the model to this stage before deployment

    Returns:
        Tuple containing paths to the sklearn and PyTorch deployment scripts and deployment info
    """
    logger.info("Creating Modal deployment scripts using templates...")

    # Check if Modal is available if deployment is requested
    if deploy and not HAS_MODAL:
        raise ImportError("Modal package not installed. Cannot deploy models.")

    # If specified, promote models to the requested stage
    if promote_to_stage:
        client = Client()
        # Get latest versions of our models
        sklearn_versions = []
        pytorch_versions = []

        all_versions = client.list_model_versions(
            model_name_or_id="iris_classification",
            hydrate=True,
        )
        for version in all_versions:
            if hasattr(version, "metadata") and version.metadata:
                # Check if run_metadata exists and has a framework attribute
                if hasattr(version.metadata, "run_metadata") and hasattr(
                    version.metadata.run_metadata, "framework"
                ):
                    if version.metadata.run_metadata.framework == "sklearn":
                        sklearn_versions.append(version)
                    elif version.metadata.run_metadata.framework == "pytorch":
                        pytorch_versions.append(version)

        # Sort by creation time (newest first)
        if sklearn_versions:
            sklearn_versions = sorted(
                sklearn_versions, key=lambda x: x.created, reverse=True
            )
            latest_sklearn = sklearn_versions[0]
            # Promote to requested stage
            sklearn_model = Model(
                name="iris_classification", version=latest_sklearn.number
            )
            sklearn_model.set_stage(stage=promote_to_stage, force=True)
            logger.info(
                f"Promoted sklearn model version {latest_sklearn.number} to {promote_to_stage}"
            )

        if pytorch_versions:
            pytorch_versions = sorted(
                pytorch_versions, key=lambda x: x.created, reverse=True
            )
            latest_pytorch = pytorch_versions[0]
            # Promote to requested stage
            pytorch_model = Model(
                name="iris_classification", version=latest_pytorch.number
            )
            pytorch_model.set_stage(stage=promote_to_stage, force=True)
            logger.info(
                f"Promoted PyTorch model version {latest_pytorch.number} to {promote_to_stage}"
            )

    # Create a temp directory for scripts to prevent cluttering workspace
    temp_dir = tempfile.mkdtemp(prefix="modal_deployment_")
    scripts_dir = Path(temp_dir)

    # Define template paths
    sklearn_template = Path("templates/sklearn_deployment_template.py")
    pytorch_template = Path("templates/pytorch_deployment_template.py")

    # Check if templates exist
    if not sklearn_template.exists():
        raise FileNotFoundError(f"sklearn template not found at {sklearn_template}")
    if not pytorch_template.exists():
        raise FileNotFoundError(f"PyTorch template not found at {pytorch_template}")

    # Define script paths with unique identifiers (for file saving, but deployment will use stage-based naming)
    sklearn_id = uuid.uuid4().hex[:8]
    pytorch_id = uuid.uuid4().hex[:8]

    sklearn_script_path = scripts_dir / f"deploy_sklearn_{sklearn_id}.py"
    pytorch_script_path = scripts_dir / f"deploy_pytorch_{pytorch_id}.py"

    # Copy the templates to the scripts directory
    shutil.copy(sklearn_template, sklearn_script_path)
    shutil.copy(pytorch_template, pytorch_script_path)

    # Make the scripts executable
    os.chmod(sklearn_script_path, 0o755)
    os.chmod(pytorch_script_path, 0o755)

    logger.info(f"Created sklearn deployment script at {sklearn_script_path}")
    logger.info(f"Created PyTorch deployment script at {pytorch_script_path}")

    # Dictionary to hold deployment information
    deployment_info = {}

    # Deploy the scripts if requested
    if deploy:
        try:
            # Deploy the sklearn model
            stage_param = f"--stage {promote_to_stage}" if promote_to_stage else ""
            sklearn_app_name = f"{app_prefix}-sklearn"
            logger.info(f"Deploying sklearn model as '{sklearn_app_name}'...")

            # Load the module containing the Modal app
            sklearn_module = load_python_module(sklearn_script_path)

            # Find the Modal app in the module
            sklearn_app = sklearn_module.app

            # Set the stage if needed
            if promote_to_stage:
                sklearn_module.MODEL_STAGE = promote_to_stage
                sklearn_module.DEPLOYMENT_ID = f"sklearn-iris-{promote_to_stage}"

            # Deploy the app using the Modal Python API
            with enable_output():
                sklearn_result = deploy_app(
                    sklearn_app, name=sklearn_app_name, environment_name="", tag=""
                )

            logger.info(f"Successfully deployed sklearn model: {sklearn_app_name}")
            deployment_info["sklearn"] = {
                "app_name": sklearn_app_name,
                "script_path": str(sklearn_script_path),
                "app_id": sklearn_result.app_id,
                "app_url": f"https://modal.com/apps/{sklearn_result.app_id}",
                "app_logs_url": sklearn_result.app_logs_url,
                "stage": promote_to_stage or "latest",
            }

            # Stream logs if requested
            if stream_logs and hasattr(sklearn_result, "app_logs_url"):
                # Note: In a real implementation, we would use Modal's streaming logs functionality
                logger.info(
                    f"Streaming logs for sklearn model from: {sklearn_result.app_logs_url}"
                )

            # Deploy the PyTorch model
            pytorch_app_name = f"{app_prefix}-pytorch"
            logger.info(f"Deploying PyTorch model as '{pytorch_app_name}'...")

            # Load the module containing the Modal app
            pytorch_module = load_python_module(pytorch_script_path)

            # Find the Modal app in the module
            pytorch_app = pytorch_module.app

            # Set the stage if needed
            if promote_to_stage:
                pytorch_module.MODEL_STAGE = promote_to_stage
                pytorch_module.DEPLOYMENT_ID = f"pytorch-iris-{promote_to_stage}"

            # Deploy the app using the Modal Python API
            with enable_output():
                pytorch_result = deploy_app(
                    pytorch_app, name=pytorch_app_name, environment_name="", tag=""
                )

            logger.info(f"Successfully deployed PyTorch model: {pytorch_app_name}")
            deployment_info["pytorch"] = {
                "app_name": pytorch_app_name,
                "script_path": str(pytorch_script_path),
                "app_id": pytorch_result.app_id,
                "app_url": f"https://modal.com/apps/{pytorch_result.app_id}",
                "app_logs_url": pytorch_result.app_logs_url,
                "stage": promote_to_stage or "latest",
            }

            # Stream logs if requested
            if stream_logs and hasattr(pytorch_result, "app_logs_url"):
                # Note: In a real implementation, we would use Modal's streaming logs functionality
                logger.info(
                    f"Streaming logs for PyTorch model from: {pytorch_result.app_logs_url}"
                )

        except Exception as e:
            logger.error(f"Error deploying to Modal: {e}")
            logger.error(traceback.format_exc())

            # Still return the script paths even if deployment failed
            deployment_info["error"] = {
                "message": str(e),
                "traceback": traceback.format_exc(),
            }

    return (str(sklearn_script_path), str(pytorch_script_path), deployment_info)


@pipeline(enable_cache=False, name="iris_model_training_and_deployment")
def train_model_pipeline(
    deploy_models: bool = False,
    stream_logs: bool = False,
    promote_to_production: bool = False,
):
    """ZenML pipeline that trains, registers, and deploys Iris classification models.

    This pipeline:
    1. Collects the active stack dependencies
    2. Trains a scikit-learn RandomForestClassifier with deployment metadata
    3. Trains a PyTorch neural network with deployment metadata
    4. Creates deployment scripts for each model using templates
    5. Optionally promotes models to production stage
    6. Optionally deploys the models to Modal using Python APIs

    Args:
        deploy_models: Whether to deploy the models to Modal
        stream_logs: Whether to stream logs from Modal deployments
        promote_to_production: Whether to promote models to production stage before deployment
    """
    stack_dependencies = get_stack_dependencies()
    train_sklearn_model(stack_dependencies=stack_dependencies)
    train_pytorch_model(stack_dependencies=stack_dependencies)

    # Determine stage for promotion if we're deploying to production
    promote_to_stage = ModelStages.PRODUCTION if promote_to_production else None

    modal_deployment(
        deploy=deploy_models,
        stream_logs=stream_logs,
        promote_to_stage=promote_to_stage,
        after=["train_sklearn_model", "train_pytorch_model"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and deploy iris classification models"
    )
    parser.add_argument(
        "--deploy", action="store_true", help="Deploy models to Modal after training"
    )
    parser.add_argument(
        "--stream-logs", action="store_true", help="Stream logs from Modal deployments"
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Promote models to production stage before deployment",
    )

    args = parser.parse_args()

    train_model_pipeline(
        deploy_models=args.deploy,
        stream_logs=args.stream_logs,
        promote_to_production=args.production,
    )
