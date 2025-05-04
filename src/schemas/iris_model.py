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

import torch


class IrisModel(torch.nn.Module):
    """PyTorch neural network for Iris classification.

    Standard torch.nn.Module implementation with explicit layers for training.
    Note: For deployment, a separate implementation exists in app/schemas.py.
    """

    def __init__(self, input_dim=4, hidden_dim=10, output_dim=3, **kwargs):
        """Initialize the IrisModel with configurable dimensions."""
        super().__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for IrisModel.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x
