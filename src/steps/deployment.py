# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import importlib.util
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from zenml import log_metadata, step

from src.utils.model_utils import (
    get_model_architecture_from_metadata,
    get_python_version_from_metadata,
)
from src.utils.yaml_config import get_config_value

try:
    import modal
    from modal import Secret

    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False
    logging.warning("Modal package not found. Deployment functionality will be limited.")

DEPLOYMENT_SCRIPT_PATH = Path(__file__).parent.parent.parent / "app" / "deployment_template.py"

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
    volume_metadata: Dict[str, str],
    environment_name: str,
    env_vars: Optional[Dict[str, str]] = None,
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """Create Modal deployment script using template and deploy it.

    Args:
        environment_name: The Modal environment to deploy to (staging, production, etc.)
        volume_metadata: Metadata about the Modal volume containing the models
        env_vars: Additional environment variables to pass to the deployment

    Returns:
        Tuple containing:
        - Path to the deployment script
        - Dictionary containing deployment information
    """
    # Check if Modal is available if deployment is requested
    if not HAS_MODAL:
        raise ImportError("Modal package not installed. Cannot deploy models.")
    if not volume_metadata:
        raise ValueError("volume_metadata is required for deployment.")

    # Deploy the scripts for both frameworks
    deployment_info: Dict[str, Dict[str, Any]] = {}
    try:
        for framework in ["sklearn", "pytorch"]:
            id_format = get_config_value(f"deployments.{framework}_id_format")
            app_name = id_format.format(stage=environment_name)
            logger.info(f"Deploying {framework} model as '{app_name}'...")

            modal_secret_name = get_config_value("modal.secret_name")

            # Update the existing secret with new environment variables if running locally
            if modal.is_local() and env_vars:
                try:
                    logger.info(f"Updating existing secret: {modal_secret_name}")

                    # Create a new secret with just the additional environment variables
                    # This will only update the keys specified in env_vars
                    # and will not affect other keys in the existing secret
                    env_updates = Secret.from_dict(env_vars)

                    # Deploy the update to the existing secret
                    env_updates.deploy(name=modal_secret_name)

                    logger.info(
                        f"Successfully updated environment variables in secret: {modal_secret_name}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to update secret: {e}")
                    logger.warning("Will continue using the existing secret without updates")

            # Set up env vars for this run
            env = {
                "MODEL_FRAMEWORK": framework,
                "MODEL_STAGE": environment_name,
                "MODAL_SECRET_NAME": modal_secret_name,
                "MODAL_VOLUME_NAME": volume_metadata["volume_name"],
                "SKLEARN_MODEL_PATH": volume_metadata["sklearn_path"],
                "PYTORCH_MODEL_PATH": volume_metadata["pytorch_path"],
            }

            python_version = get_python_version_from_metadata()
            env["PYTHON_VERSION"] = python_version

            if framework == "pytorch":
                architecture = get_model_architecture_from_metadata()
                env["MODEL_ARCHITECTURE"] = json.dumps(architecture)

            if env_vars:
                env.update(env_vars)

            # Load the module containing the Modal app
            module = load_python_module(str(DEPLOYMENT_SCRIPT_PATH))

            # The unified app handles both frameworks
            deploy_result, fastapi_url = module.run_deployment_entrypoint(
                framework=framework,
                stage=environment_name,
                volume=volume_metadata["volume_name"],
                model_path=volume_metadata[f"{framework}_path"],
                python_version=python_version,
            )

            logger.info(f"Successfully deployed {framework} model: {app_name}")

            deployment_info[framework] = {
                "app_name": app_name,
                "app_id": deploy_result.app_id,
                "app_logs_url": getattr(deploy_result, "app_logs_url", None),
                "fastapi_url": fastapi_url,
                "stage": "latest",
                "volume_info": volume_metadata,
                "env_vars": list(env.keys()),
            }

        # Single metadata log with all deployment information
        log_metadata(
            metadata={
                "deployment": {
                    "pytorch_info": deployment_info["pytorch"],
                    "sklearn_info": deployment_info["sklearn"],
                    "script_path": str(DEPLOYMENT_SCRIPT_PATH),
                    "volume_metadata": volume_metadata,
                }
            }
        )

    except Exception as e:
        logger.error(f"Error deploying to Modal: {e}")
        logger.error(traceback.format_exc())

        # Still return the script path even if deployment failed
        deployment_info["error"] = {
            "message": str(e),
            "traceback": traceback.format_exc(),
        }

    return (str(DEPLOYMENT_SCRIPT_PATH), deployment_info)
