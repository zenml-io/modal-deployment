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

from zenml import pipeline

from src.steps import (
    save_to_modal_volume,
    train_pytorch_model,
    train_sklearn_model,
)


@pipeline
def train_model_pipeline() -> None:
    """Trains, registers, and deploys Iris classification models in a ZenML pipeline.

    This pipeline:
    1. Collects the active stack dependencies
    2. Trains a scikit-learn RandomForestClassifier with deployment metadata
    3. Trains a PyTorch neural network with deployment metadata
    4. Creates deployment scripts for each model using templates
    5. Optionally promotes models to production stage
    6. Optionally deploys the models to Modal using Python APIs

    Args:
        deploy_models: Whether to deploy the models to Modal
        stream_logs: Whether to stream logs from Modal deployments
        promote_to_production: Whether to promote models to production stage before deployment
        environment: The Modal environment to deploy to (staging, production, etc.)
    """
    sklearn_model = train_sklearn_model()
    pytorch_model = train_pytorch_model()

    save_to_modal_volume(
        sklearn_model=sklearn_model,
        pytorch_model=pytorch_model,
    )
