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
from zenml import step

from src.utils.yaml_config import get_config_value


@step
def save_to_modal_volume(
    sklearn_model: Annotated[RandomForestClassifier, "sklearn_model"],
    pytorch_model: Annotated[torch.nn.Module, "pytorch_model"],
    app_prefix: Optional[str] = None,
    modal_secret_name: Optional[str] = None,
) -> dict:
    """Dump both models into a Modal volume and return volume metadata."""
    # 1. get config values
    if app_prefix is None:
        app_prefix = get_config_value("deployments.app_prefix")
    if modal_secret_name is None:
        modal_secret_name = get_config_value("modal.secret_name")

    # 2. create/get the volume
    volume_name = f"{app_prefix}-models"
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
        image=modal.Image.debian_slim().pip_install("torch", "pickle"),
        mounts={"/models": volume},
        secrets=[modal.Secret.from_name(modal_secret_name)],
    )
    def _cp(sk_src: str, pt_src: str):
        os.makedirs("/models", exist_ok=True)
        shutil.copy(sk_src, "/models/sklearn_model.pkl")
        shutil.copy(pt_src, "/models/pytorch_model.pth")

    _cp.call(str(sk_path), str(pt_path))

    return {
        "volume_name": volume_name,
        "sklearn_path": "/models/sklearn_model.pkl",
        "pytorch_path": "/models/pytorch_model.pth",
    }
