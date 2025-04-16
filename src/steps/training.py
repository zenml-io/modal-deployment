import datetime
import logging
import platform
from typing import Annotated, List

import torch
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from zenml import get_step_context, log_metadata, step
from zenml.client import Client
from zenml.integrations.registry import integration_registry

logger = logging.getLogger("zenml_deployment")


# Define a simple neural network model
class IrisModel(torch.nn.Module):
    """PyTorch neural network for Iris classification."""

    def __init__(self):
        super(IrisModel, self).__init__()
        self.layer1 = torch.nn.Linear(4, 10)
        self.layer2 = torch.nn.Linear(10, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for IrisModel.

        Args:
            x: Input tensor of shape (batch_size, 4).

        Returns:
            Output tensor of shape (batch_size, 3).
        """
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


def log_stack_dependencies(
    modal_secret_name: str,
) -> Annotated[List[str], "dependencies"]:
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
        # get the current model
        mv = get_step_context().model
        current_model_name = mv.name

        # Look for existing versions of our model
        model_versions = client.list_model_versions(model_name_or_id=current_model_name)

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
                    "modal_secret": modal_secret_name,
                }
            }

            # Log the updated metadata using the zenml log_metadata function
            log_metadata(
                metadata=deployment_metadata,
                model_name=current_model_name,
                model_version=latest_version.number,
            )
            logger.info(
                f"Logged dependencies to {current_model_name} model version {latest_version.number}"
            )
    except Exception as e:
        logger.warning(f"Could not log dependencies to existing model: {e}")
        logger.info("Dependencies will be logged during model training steps")

    return unique_deps


@step
def train_sklearn_model(
    modal_secret_name: str,
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

    stack_dependencies = log_stack_dependencies(modal_secret_name)

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
                "modal_secret": modal_secret_name,
                "python_version": python_version,
            },
        },
        infer_model=True,
    )

    # Get the current model - it will already be in "latest" stage by default
    current_model = get_step_context().model
    logger.info(
        f"Registered iris-classification sklearn model as version {current_model.version}"
    )

    return model


@step
def train_pytorch_model(
    modal_secret_name: str,
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

    stack_dependencies = log_stack_dependencies(modal_secret_name)
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
                "modal_secret": modal_secret_name,
                "architecture": architecture,
                "python_version": python_version,
            },
        },
        infer_model=True,
    )

    # Get the current model - it will already be in "latest" stage by default
    current_model = get_step_context().model
    logger.info(
        f"Registered iris-classification pytorch model as version {current_model.version}"
    )

    return model
