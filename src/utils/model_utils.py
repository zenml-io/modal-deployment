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

from typing import Any, Dict, Optional

from zenml.client import Client
from zenml.models.v2.core.model_version import ModelVersionResponse

from src.utils.constants import MODEL_NAME


def find_model_version(
    framework: Optional[str] = None,
) -> ModelVersionResponse:
    """Find the most recent model version, optionally filtering by framework."""
    client = Client()
    all_versions = client.list_model_versions(
        model_name_or_id=MODEL_NAME,
        hydrate=True,
    )
    if not all_versions:
        raise ValueError(f"No model versions found for {MODEL_NAME}")

    # Sort newest first
    all_versions.sort(key=lambda v: v.created, reverse=True)

    if framework:
        filtered = [
            v
            for v in all_versions
            if getattr(v.metadata, "run_metadata", None)
            and getattr(v.metadata.run_metadata, "framework", None) == framework
        ]
        if filtered:
            print(f"Using latest {framework} version: {filtered[0].number}")
            return filtered[0]
        print(
            f"No '{framework}' versions found; falling back to latest: {all_versions[0].number}"
        )

    # either no framework requested or no match found
    return all_versions[0]


def get_python_version_from_metadata(
    framework: Optional[str] = None,
    default: str = "3.10",
) -> str:
    """Get the Python version from metadata of the latest (optionally framework-filtered) model."""
    version_meta = find_model_version(framework).metadata or {}
    deployment = getattr(version_meta, "deployment", None)
    python_ver = getattr(deployment, "python_version", None)
    if python_ver:
        print(f"Found Python version in metadata: {python_ver}")
        return python_ver
    print(f"Using default Python version: {default}")
    return default


def get_model_architecture_from_metadata(
    framework: str = "pytorch",
    default: Dict[str, Any] = {"input_dim": 4, "hidden_dim": 10, "output_dim": 3},
) -> Dict[str, Any]:
    """Get architecture for the latest model version of `framework`."""
    model_version = find_model_version(framework)
    deployment_metadata = model_version.metadata or {}
    architecture = getattr(deployment_metadata, "architecture", None)
    if architecture:
        print("Found architecture in deployment metadata")
        return architecture

    print(f"Using default architecture: {default}")
    return default
