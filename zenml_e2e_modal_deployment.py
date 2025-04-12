from zenml import step, pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from zenml import Model, log_metadata, get_step_context
from rich import print
from typing import Annotated, Union, Tuple, Optional
import torch
import os
import tempfile
import shutil
import docker
import logging
import uuid
import datetime

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
def build_docker_image(
    sklearn_model: Annotated[RandomForestClassifier, "sklearn_model"],
    pytorch_model: Annotated[torch.nn.Module, "pytorch_model"],
) -> Annotated[str, "image_name"]:
    """Build a Docker image for the ZenML-based model server."""
    logger.info("Building Docker image for model server...")

    # Create a unique image tag
    image_tag = f"iris-models:{uuid.uuid4().hex[:8]}"

    # Create a temporary directory for the Docker build context
    with tempfile.TemporaryDirectory() as build_dir:
        logger.info(f"Created temporary build directory: {build_dir}")

        # Copy app files to build directory
        app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
        if os.path.exists(app_dir):
            for file_name in os.listdir(app_dir):
                src_path = os.path.join(app_dir, file_name)
                dst_path = os.path.join(build_dir, file_name)
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)
                    logger.info(f"Copied {src_path} to {dst_path}")
        else:
            raise FileNotFoundError(f"App directory not found: {app_dir}")

        # Copy this script file to the build directory
        script_path = os.path.abspath(__file__)
        dst_script_path = os.path.join(build_dir, os.path.basename(script_path))
        shutil.copy2(script_path, dst_script_path)
        logger.info(f"Copied {script_path} to {dst_script_path}")

        # No need to save models directly anymore - they'll be loaded from ZenML

        # Build the Docker image
        try:
            docker_client = docker.from_env()

            # Build the image
            logger.info(f"Building Docker image: {image_tag}")
            image, logs = docker_client.images.build(
                path=build_dir,
                tag=image_tag,
                rm=True,
                forcerm=True,
            )

            # Print build logs
            for log in logs:
                if "stream" in log:
                    log_line = log["stream"].strip()
                    if log_line:
                        logger.info(f"Docker build: {log_line}")

            logger.info(f"Successfully built Docker image: {image_tag}")

            # Log the built image information as metadata to both models
            for model_name in ["sklearn_model", "pytorch_model"]:
                log_metadata(
                    metadata={
                        "deployment": {
                            "image_tag": image_tag,
                            "built_at": str(datetime.datetime.now()),
                            "includes": ["FastAPI", "ZenML", "sklearn", "PyTorch"],
                        }
                    },
                    model_name=model_name,
                    model_version="production",
                )

            return image_tag

        except docker.errors.BuildError as e:
            logger.error(f"Docker build error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error building Docker image: {e}")
            raise


@step(enable_cache=False)
def run_deployment_container(
    image_name: str,
    zenml_server_url: str,
    zenml_api_key: str,
    sklearn_model_name: str = "sklearn_model",
    sklearn_model_stage: str = "production",
    pytorch_model_name: str = "pytorch_model",
    pytorch_model_stage: str = "production",
    host_port: int = 8000,
) -> Tuple[str, str]:
    """Run the Docker container locally, passing env vars to connect with ZenML."""
    logger.info(f"Running deployment container from image: {image_name}")

    client = docker.from_env()

    # Container naming
    container_name = f"iris-predictor-{uuid.uuid4().hex[:8]}".lower()

    # Environment variables for the container
    env_vars = {
        "ZENML_STORE_URL": zenml_server_url,
        "ZENML_STORE_API_KEY": zenml_api_key,
        "SKLEARN_MODEL_NAME": sklearn_model_name,
        "SKLEARN_MODEL_STAGE": sklearn_model_stage,
        "PYTORCH_MODEL_NAME": pytorch_model_name,
        "PYTORCH_MODEL_STAGE": pytorch_model_stage,
    }

    # Port mapping
    ports = {f"8000/tcp": host_port}

    # Remove existing container if it exists
    try:
        existing = client.containers.get(container_name)
        logger.info(f"Stopping existing container: {container_name}")
        existing.stop()
        existing.remove()
    except docker.errors.NotFound:
        pass

    # Run the container
    try:
        logger.info(f"Starting container: {container_name}")
        container = client.containers.run(
            image=image_name,
            name=container_name,
            environment=env_vars,
            ports=ports,
            detach=True,
            restart_policy={"Name": "unless-stopped"},
        )

        container_id = container.id
        service_url = f"http://localhost:{host_port}"

        logger.info(f"Started container {container_name} with ID {container_id}")
        logger.info(f"Service URL: {service_url}")

        # Log deployment information as metadata
        for model_name in ["sklearn_model", "pytorch_model"]:
            log_metadata(
                metadata={
                    "deployment_status": {
                        "container_id": container_id,
                        "container_name": container_name,
                        "service_url": service_url,
                        "deployed_at": str(datetime.datetime.now()),
                        "status": "running",
                    }
                },
                model_name=model_name,
                model_version="production",
            )

        return container_id, service_url
    except Exception as e:
        logger.error(f"Failed to start container: {e}")
        raise


@step
def print_deployment_instructions(image_name: str) -> None:
    """Print instructions for running the Docker container manually."""
    print(f"Built Docker image: {image_name}")
    print("\nTo run the container manually:")
    print(f"""docker run -d \\
  -p 8000:8000 \\
  -e ZENML_STORE_URL="your-zenml-server-url" \\
  -e ZENML_STORE_API_KEY="your-zenml-api-key" \\
  -e SKLEARN_MODEL_NAME="sklearn_model" \\
  -e SKLEARN_MODEL_STAGE="production" \\
  -e PYTORCH_MODEL_NAME="pytorch_model" \\
  -e PYTORCH_MODEL_STAGE="production" \\
  --name iris-predictor \\
  {image_name}""")

    print("\nOnce running, test with:")
    print("""curl -X POST http://localhost:8000/predict/sklearn \\
  -H "Content-Type: application/json" \\
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'""")


@step(enable_cache=False)
def deploy_container_if_possible(
    image_name: str,
) -> Optional[Tuple[str, str]]:
    """Attempt to deploy the container if ZenML environment variables are available."""
    zenml_url = os.environ.get("ZENML_STORE_URL")
    zenml_api_key = os.environ.get("ZENML_STORE_API_KEY")

    if zenml_url and zenml_api_key:
        logger.info("ZenML environment variables found, deploying container...")
        container_id, service_url = run_deployment_container(
            image_name=image_name,
            zenml_server_url=zenml_url,
            zenml_api_key=zenml_api_key,
            sklearn_model_name="sklearn_model",
            pytorch_model_name="pytorch_model",
        )
        print(f"Deployed service at: {service_url}")
        print("To test the sklearn model:")
        print("""curl -X POST http://localhost:8000/predict/sklearn \\
  -H "Content-Type: application/json" \\
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'""")
        print("\nTo test the PyTorch model:")
        print("""curl -X POST http://localhost:8000/predict/pytorch \\
  -H "Content-Type: application/json" \\
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'""")
        return container_id, service_url
    else:
        logger.info("ZenML environment variables not found, skipping deployment")
        return None


@pipeline(enable_cache=False)
def train_model_pipeline():
    sklearn_model = train_sklearn_model()
    pytorch_model = train_pytorch_model()
    image_name = build_docker_image(sklearn_model, pytorch_model)
    print_deployment_instructions(image_name)
    deploy_container_if_possible(image_name)


if __name__ == "__main__":
    train_model_pipeline()
