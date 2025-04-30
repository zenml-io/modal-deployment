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

from zenml import pipeline

from src.steps import (
    load_volume_metadata_from_model,
    modal_deployment,
)
from src.utils.yaml_config import get_config_value


@pipeline
def deploy_model_pipeline(
    environment: str = "staging",
    stream_logs: bool = False,
    volume_metadata: Optional[Dict[str, Any]] = None,
):
    """Deploy the model to Modal."""
    volume_metadata = load_volume_metadata_from_model()

    modal_deployment(
        app_prefix=get_config_value("deployments.app_prefix"),
        environment_name=environment,
        volume_metadata=volume_metadata,
        stream_logs=stream_logs,
    )
