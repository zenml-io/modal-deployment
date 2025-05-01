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


@step
def load_volume_metadata_from_pipeline_run() -> Annotated[Dict[str, Any], "volume_metadata"]:
    """Load volume metadata from the latest pipeline run."""
    client = Client()
    pipeline = client.get_pipeline("train_model_pipeline")
    latest_run = pipeline.last_run
    run_metadata = latest_run.run_metadata

    step_metadata_key = "save_to_modal_volume::deployment"
    volume_metadata = run_metadata.get(step_metadata_key, {}).get("volume_metadata")
    if not volume_metadata:
        raise RuntimeError(f"No volume_metadata found in pipeline run '{latest_run.id}'")
    return volume_metadata
