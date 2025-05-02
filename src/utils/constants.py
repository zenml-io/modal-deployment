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

"""Constants module with fixed values and values loaded from config files."""

import os

from src.utils.yaml_config import get_config_value

# Map prediction indices to species names
SPECIES_MAPPING = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}

# Model stage can be set via environment variable or defaults to 'staging'
MODEL_NAME = get_config_value("model.name")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "staging")

# Load Modal values from configuration
MODAL_SECRET_NAME = get_config_value("modal.secret_name")
TRAINING_PIPELINE_NAME = "train_model_pipeline"
