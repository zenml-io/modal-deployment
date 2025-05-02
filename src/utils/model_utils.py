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

import logging
from typing import Any, Dict

import modal
from zenml.client import Client
from zenml.enums import ExecutionStatus

from src.utils.constants import TRAINING_PIPELINE_NAME

logger = logging.getLogger(__name__)


def find_latest_run_metadata(
    framework: str = "sklearn",
    step_name: str = "train_sklearn_model",
) -> Dict[str, Any]:
    """Find and return the run metadata from the most recent successful run of a pipeline.

    This function searches for the latest completed run of the specified pipeline
    where the given step used the specified ML framework.

    Args:
        framework: ML framework name to filter runs by (e.g., "pytorch", "tensorflow")
        step_name: Name of the step to check for framework usage, defaults to "train_pytorch_model"

    Returns:
        Dict[str, Any]: Metadata dictionary from the matching pipeline run

    Raises:
        RuntimeError: If no completed runs match the specified criteria
    """
    client = Client()

    # 1) look up the pipeline
    pipeline = client.get_pipeline(TRAINING_PIPELINE_NAME)

    # 2) page through its runs, newest first
    runs = client.list_pipeline_runs(
        pipeline_id=pipeline.id,
        status=ExecutionStatus.COMPLETED,
        sort_by="desc:created",
        hydrate=True,
    )

    for run in runs:
        md = run.run_metadata or {}
        # each step logs a key like "train_pytorch_model::framework"
        key = f"{step_name}::framework"
        if md.get(key) == framework:
            return md

    raise RuntimeError(
        f"No completed runs of pipeline '{TRAINING_PIPELINE_NAME}' found with "
        f"{step_name}::framework == '{framework}'"
    )


def get_python_version_from_metadata(
    framework: str = "sklearn",
    default: str = "3.10",
) -> str:
    """Get the Python version from metadata of the latest (optionally framework-filtered) model."""
    step_name = f"train_{framework}_model"

    version_meta = (
        find_latest_run_metadata(
            framework=framework,
            step_name=step_name,
        )
        or {}
    )

    python_ver = version_meta[f"{step_name}::python_version"]
    if python_ver:
        print(f"Found Python version in metadata: {python_ver}")
        return python_ver
    print(f"Using default Python version: {default}")
    return default


def get_model_architecture_from_metadata(
    framework: str = "pytorch",
    default: Dict[str, Any] = {
        "input_dim": 4,
        "hidden_dim": 10,
        "output_dim": 3,
    },
) -> Dict[str, Any]:
    """Get architecture for the latest model version of `framework`."""
    step_name = f"train_{framework}_model"

    version_meta = (
        find_latest_run_metadata(
            framework=framework,
            step_name=step_name,
        )
        or {}
    )

    architecture = version_meta[f"{step_name}::architecture"]
    if architecture:
        print("Found architecture in deployment metadata")
        return architecture

    print(f"Using default architecture: {default}")
    return default


def check_models_exist(environment: str) -> bool:
    """Check if models exist in the specified Modal volume."""
    volume_mapping = {
        "staging": "iris-staging-models",
        "production": "iris-prod-models",
    }

    # Get the correct volume name based on environment
    volume_name = volume_mapping.get(environment)

    if not volume_name:
        logging.error(f"Unknown environment: {environment}")
        return False

    required_files = [
        "sklearn_model.pkl",
        "pytorch_model.pth",
    ]

    try:
        vol = modal.Volume.from_name(volume_name, environment_name=environment)
        logger.info(
            f"Checking for models in volume: {volume_name}, environment: {environment}"
        )

        files_in_volume = vol.listdir("/")
        file_paths = [entry.path for entry in files_in_volume]

        # Check if required files exist
        missing_files = []
        for file_path in required_files:
            if file_path not in file_paths:
                missing_files.append(file_path)

        if missing_files:
            logger.error(f"Missing files: {missing_files}")
            return False

        return True

    except Exception as e:
        logger.error(f"Error checking volume '{volume_name}': {e}")
        return False
