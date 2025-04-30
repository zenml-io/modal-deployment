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

from typing import Annotated, Any, Dict

from zenml import step
from zenml.client import Client

from src.utils.yaml_config import get_config_value


@step
def load_volume_metadata_from_model() -> Annotated[
    Dict[str, Any], "volume_metadata"
]:
    """Load volume metadata from the latest trainingpipeline run."""
    client = Client()
    model_name = get_config_value("model.name")
    versions = client.list_model_versions(model_name_or_id=model_name, hydrate=True)

    if not versions:
        raise RuntimeError(f"No versions found for model '{model_name}'")

    latest = sorted(versions, key=lambda v: v.created, reverse=True)[0]

    # stored under metadata.deployment.volume_metadata
    deployment_meta = getattr(latest.metadata, "deployment", {}) or {}
    volume_metadata = deployment_meta.get("volume_metadata")
    if not volume_metadata:
        raise RuntimeError(
            f"No volume_metadata found in model '{model_name}' version {latest.number}"
        )
    return volume_metadata
