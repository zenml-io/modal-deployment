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
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

from zenml import log_metadata, step

from src.utils.model_utils import get_model_architecture_from_metadata

try:
    from modal import Volume
    from modal.output import enable_output
    from modal.runner import deploy_app

    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False
    logging.warning(
        "Modal package not found. Deployment functionality will be limited."
    )

DEPLOYMENT_SCRIPT_PATH = (
    Path(__file__).parent.parent / "templates" / "deployment_template.py"
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
    volume_metadata: Dict[str, str],
    environment_name: str = "staging",
    stream_logs: bool = False,
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """Create Modal deployment script using template and deploy it.

    Args:
        stream_logs: Whether to stream logs from Modal deployments
        environment_name: The Modal environment to deploy to (staging, production, etc.)
        volume_metadata: Metadata about the Modal volume containing the models

    Returns:
        Tuple containing:
        - Path to the deployment script
        - Dictionary containing deployment information
    """
    from src.utils.yaml_config import get_config_value

    # Check if Modal is available if deployment is requested
    if not HAS_MODAL:
        raise ImportError("Modal package not installed. Cannot deploy models.")
    if not volume_metadata:
        raise ValueError("volume_metadata is required for deployment.")

    # build volume mounts
    volume = Volume.from_name(volume_metadata["volume_name"])
    volumes = {"/models": volume}

    # Deploy the scripts for both frameworks
    deployment_info: Dict[str, Dict[str, Any]] = {}
    try:
        import json

        for framework in ["sklearn", "pytorch"]:
            id_format = get_config_value(f"deployments.{framework}_id_format")
            app_name = id_format.format(stage=environment_name)
            logger.info(f"Deploying {framework} model as '{app_name}'...")

            # Set up env vars for this run
            env = {
                "MODEL_FRAMEWORK": framework,
                "MODEL_STAGE": environment_name,
                "MODAL_SECRET_NAME": get_config_value("modal.secret_name"),
                "MODAL_VOLUME_NAME": volume_metadata["volume_name"],
                "SKLEARN_MODEL_PATH": volume_metadata["sklearn_path"],
                "PYTORCH_MODEL_PATH": volume_metadata["pytorch_path"],
            }

            if framework == "pytorch":
                architecture = get_model_architecture_from_metadata()
                env["MODEL_ARCHITECTURE"] = json.dumps(architecture)

            # Load the module containing the Modal app
            module = load_python_module(str(DEPLOYMENT_SCRIPT_PATH))

            # The unified app handles both frameworks
            app = module.create_modal_app(
                framework=framework,
                stage=environment_name,
                volume_name=volume_metadata["volume_name"],
            )

            # Deploy the app using the Modal Python API with environment variables
            with enable_output():
                result = deploy_app(
                    app,
                    name=app_name,
                    environment_name=environment_name,
                    tag="",
                    env=env,
                    volumes=volumes,
                )

            logger.info(f"Successfully deployed {framework} model: {app_name}")

            deployment_info[framework] = {
                "app_name": app_name,
                "app_id": result.app_id,
                "app_logs_url": getattr(result, "app_logs_url", None),
                "stage": "latest",
                "volume_info": volume_metadata,
            }

            # Stream logs if requested
            if stream_logs and hasattr(result, "app_logs_url"):
                logger.info(
                    f"Streaming logs for {framework} model from: {result.app_logs_url}"
                )

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
