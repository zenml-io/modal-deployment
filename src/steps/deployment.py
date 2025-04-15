import importlib.util
import logging
import os
import shutil
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from zenml import Model, get_step_context, step
from zenml.client import Client

try:
    from modal.output import enable_output
    from modal.runner import deploy_app

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


def load_python_module(file_path: str) -> Any:
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
    environment_name: str = "staging",
) -> Tuple[str, str, Dict[str, Dict[str, Any]]]:
    """Create Modal deployment scripts using templates and optionally deploy them.

    Args:
        deploy: Whether to actually deploy the scripts using Modal
        stream_logs: Whether to stream logs from the deployments
        app_prefix: Prefix to use for app names
        promote_to_stage: If specified, promote the model to this stage before deployment
        environment_name: The Modal environment to deploy to (staging, production, etc.)

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

        # get the current model
        mv = get_step_context().model
        current_model_name = mv.name

        all_versions = client.list_model_versions(
            model_name_or_id=current_model_name,
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
                name=current_model_name, version=latest_sklearn.number
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
                name=current_model_name, version=latest_pytorch.number
            )
            pytorch_model.set_stage(stage=promote_to_stage, force=True)
            logger.info(
                f"Promoted PyTorch model version {latest_pytorch.number} to {promote_to_stage}"
            )

    # Create a temp directory for scripts to prevent cluttering workspace
    temp_dir = tempfile.mkdtemp(prefix="modal_deployment_")
    scripts_dir = Path(temp_dir)

    # Define template paths
    sklearn_template = (
        Path(__file__).parent.parent / "templates" / "sklearn_deployment_template.py"
    )
    pytorch_template = (
        Path(__file__).parent.parent / "templates" / "pytorch_deployment_template.py"
    )

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
            sklearn_app_name = f"{app_prefix}-sklearn"
            logger.info(f"Deploying sklearn model as '{sklearn_app_name}'...")

            # Load the module containing the Modal app
            sklearn_module = load_python_module(sklearn_script_path)

            # Find the Modal app in the module
            sklearn_app = sklearn_module.app

            # Set the stage if needed
            if promote_to_stage:
                sklearn_module.MODEL_STAGE = promote_to_stage
                sklearn_module.SKLEARN_DEPLOYMENT_ID = (
                    f"sklearn-iris-{promote_to_stage}"
                )

            # Deploy the app using the Modal Python API
            with enable_output():
                sklearn_result = deploy_app(
                    sklearn_app,
                    name=sklearn_app_name,
                    environment_name=environment_name,
                    tag="",
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
                pytorch_module.PYTORCH_DEPLOYMENT_ID = (
                    f"pytorch-iris-{promote_to_stage}"
                )

            # Deploy the app using the Modal Python API
            with enable_output():
                pytorch_result = deploy_app(
                    pytorch_app,
                    name=pytorch_app_name,
                    environment_name=environment_name,
                    tag="",
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
