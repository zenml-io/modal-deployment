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

import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Optional

import modal
import torch
from sklearn.ensemble import RandomForestClassifier
from zenml import log_metadata, step

from src.utils.yaml_config import get_config_value


@step
def save_to_modal_volume(
    sklearn_model: Annotated[RandomForestClassifier, "sklearn_model"],
    pytorch_model: Annotated[torch.nn.Module, "pytorch_model"],
    app_prefix: Optional[str] = None,
    modal_secret_name: Optional[str] = None,
    volume_name: Optional[str] = None,
    sklearn_path: Optional[str] = None,
    pytorch_path: Optional[str] = None,
) -> Annotated[dict, "volume_metadata"]:
    """Dump both models into a Modal volume and return volume metadata."""
    # 1. get config values
    app_prefix = app_prefix or get_config_value("deployments.app_prefix")
    modal_secret_name = modal_secret_name or get_config_value("modal.secret_name")

    # 2. define our fixed names in code
    volume_name = volume_name or f"{app_prefix}-models"
    sklearn_path = sklearn_path or "/models/sklearn_model.pkl"
    pytorch_path = pytorch_path or "/models/pytorch_model.pth"

    # 2. create/get the volume
    volume = modal.Volume.from_name(volume_name)

    # 3. write models to a temp dir
    tmp = Path(tempfile.mkdtemp())
    sk_path = tmp / "sklearn_model.pkl"
    with open(sk_path, "wb") as f:
        pickle.dump(sklearn_model, f)
    pt_path = tmp / "pytorch_model.pth"
    torch.save(pytorch_model.state_dict(), pt_path)

    # 4. copy into the volume via Modal
    @modal.function(
        image=modal.Image.debian_slim(),
        mounts={"/models": volume},
        secrets=[modal.Secret.from_name(modal_secret_name)],
    )
    def _cp(sk_src: str, pt_src: str, sk_dest: str, pt_dest: str):
        os.makedirs(os.path.dirname(sk_dest), exist_ok=True)
        shutil.copy(sk_src, sklearn_path)
        shutil.copy(pt_src, pytorch_path)
        return {"sklearn": sk_dest, "pytorch": pt_dest}

    _cp.call(
        str(sk_path),
        str(pt_path),
        sklearn_path,
        pytorch_path,
    )

    volume_metadata = {
        "volume_name": volume_name,
        "sklearn_path": sklearn_path,
        "pytorch_path": pytorch_path,
    }

    log_metadata(
        metadata={
            "deployment": {
                "volume_metadata": volume_metadata,
            }
        },
        infer_model=True,
    )

    return volume_metadata
