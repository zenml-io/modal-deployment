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

from typing import List

from pydantic import BaseModel, Field


class IrisModel:
    """PyTorch model for Iris classification."""

    def __init__(self, input_dim=4, hidden_dim=10, output_dim=3):
        """Initialize model architecture."""
        try:
            import torch.nn as nn

            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
        except ImportError:
            # This will only be used when framework == "pytorch"
            pass

    def __call__(self, x):
        """Forward pass."""
        return self.model(x)

    def load_state_dict(self, state_dict):
        """Load model weights."""
        try:
            import torch

            self.model.load_state_dict(state_dict)
        except ImportError:
            # This will only be used when framework == "pytorch"
            pass

    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self


class IrisFeatures(BaseModel):
    """Request model for iris prediction."""

    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)


class PredictionResponse(BaseModel):
    """Response model for iris prediction."""

    prediction: int
    prediction_probabilities: List[float]
    species_name: str


class ApiEndpoints(BaseModel):
    """URLs for the API endpoints."""

    root: str
    health: str
    predict: str
    url: str


class ApiInfo(BaseModel):
    """Response model for the root endpoint."""

    message: str
    deployment_id: str
    endpoints: ApiEndpoints
    model: str
    framework: str
    stage: str
    timestamp: str
