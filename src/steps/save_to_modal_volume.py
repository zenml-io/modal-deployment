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

import pickle
import tempfile
from pathlib import Path
from typing import Annotated

import torch
from modal import Volume
from sklearn.ensemble import RandomForestClassifier
from zenml import log_metadata, step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step(enable_cache=False)
def save_to_modal_volume(
    sklearn_model: Annotated[RandomForestClassifier, "sklearn_model"],
    pytorch_model: Annotated[torch.nn.Module, "pytorch_model"],
    environment_name: str,  # set in yaml config
    volume_name: str,  # set in yaml config
    sklearn_path: str,  # set in yaml config
    pytorch_path: str,  # set in yaml config
) -> Annotated[dict, "volume_metadata"]:
    """Dump both models into a Modal volume and return volume metadata."""
    # 1. log info
    logger.info(f"Saving models to Modal volume: {volume_name}")
    logger.info(f"Sklearn model path: {sklearn_path}")
    logger.info(f"Pytorch model path: {pytorch_path}")

    # 2. write models to a temp dir
    tmp = Path(tempfile.mkdtemp())
    sk_file = tmp / "sklearn_model.pkl"
    with open(sk_file, "wb") as f:
        pickle.dump(sklearn_model, f)
    pt_file = tmp / "pytorch_model.pth"
    torch.save(pytorch_model.state_dict(), pt_file)

    # 3. upload directly into the Modal Volume from the host
    vol = Volume.from_name(
        volume_name,
        create_if_missing=True,
        environment_name=environment_name,
    )

    # check if the files exist in the volume before putting them
    # if they do, delete them first
    files_in_volume = vol.listdir("/models")
    for file_entry in files_in_volume:
        if file_entry.path == sklearn_path or file_entry.path == pytorch_path:
            logger.info(f"Deleting existing file: {file_entry.path}")
            vol.remove_file(file_entry.path)

    # upload the new files
    with vol.batch_upload() as batch:
        batch.put_file(str(sk_file), sklearn_path)
        batch.put_file(str(pt_file), pytorch_path)

    # 4. create volume metadata
    volume_metadata = {
        "environment_name": environment_name,
        "volume_name": volume_name,
        "sklearn_path": sklearn_path,
        "pytorch_path": pytorch_path,
    }

    # 5. log metadata to the model
    log_metadata(
        metadata={
            "deployment": {
                "volume_metadata": volume_metadata,
            }
        },
        infer_model=True,
    )

    # 6. log metadata to pipeline run also
    log_metadata(
        metadata={
            "deployment": {
                "volume_metadata": volume_metadata,
            }
        },
    )

    return volume_metadata
