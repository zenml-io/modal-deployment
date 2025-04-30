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

"""Configuration manager for YAML files with merge-key and variable interpolation support."""

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Cache for loaded configurations
_cache: Dict[str, Dict[str, Any]] = {}

# Pattern for ${var} interpolation
_INTERP = re.compile(r"\$\{([^}]+)\}")


def _interpolate(obj: Any, config: Dict[str, Any]) -> Any:
    """Recursively replace '${a.b}' in strings using values from config or env.

    Args:
        obj: The object to interpolate.
        config: The config to use for interpolation.

    Returns:
        The interpolated object.
    """
    if isinstance(obj, dict):
        return {k: _interpolate(v, config) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(v, config) for v in obj]
    if isinstance(obj, str):

        def repl(m):
            path = m.group(1)
            # env var?
            if "." not in path and path in os.environ:
                return os.environ[path]
            # config lookup
            cur = config
            for p in path.split("."):
                if not isinstance(cur, dict) or p not in cur:
                    return m.group(0)
                cur = cur[p]
            return str(cur)

        return _INTERP.sub(repl, obj)
    return obj


def get_config(
    prefix: str = "common",
    environment: Optional[str] = None,
    force_reload: bool = False,
) -> Dict[str, Any]:
    """Load and cache a merged configuration dict from YAML with imports & interpolation.

    Args:
        prefix: The prefix of the config file to load.
        environment: The environment to load the config for.
        force_reload: Whether to force reload the config.

    Returns:
        The merged configuration dict.
    """
    key = f"{prefix}_{environment}" if environment else prefix
    if not force_reload and key in _cache:
        return _cache[key]

    BASE_DIR = Path(__file__).parent.parent  # this is "src/"
    CONFIG_DIR = BASE_DIR / "configs"

    fname = f"{prefix}_{environment}.yaml" if environment else f"{prefix}.yaml"
    path = CONFIG_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    # First load common config to get the COMMON anchor
    common_path = CONFIG_DIR / "common.yaml"
    if prefix != "common" and common_path.exists():
        # Load both files into a single stream with common first
        combined_yaml = common_path.read_text() + "\n" + path.read_text()
        raw = yaml.safe_load(combined_yaml) or {}
    else:
        raw = yaml.safe_load(path.read_text()) or {}

    merged = _interpolate(raw, raw)
    _cache[key] = merged
    return merged


def get_merged_config_path(prefix: str, environment: str) -> str:
    """Get a path to a temporary file containing the merged configuration.

    This creates a temporary file with the fully merged and interpolated
    configuration that ZenML can load directly.

    Args:
        prefix: The configuration prefix (train or deploy)
        environment: The environment (staging or production)

    Returns:
        Path to a temporary file containing the merged configuration
    """
    # Get the merged configuration
    merged_cfg = get_config(prefix, environment)

    # Create a temporary file with the merged configuration
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(merged_cfg, tmp)
        tmp.flush()
        return tmp.name


def get_config_value(
    key_path: str,
    prefix: str = "common",
    environment: Optional[str] = None,
    default: Any = None,
) -> Any:
    """Retrieve a nested config value by dot-separated key path.

    Args:
        key_path: The dot-separated key path to retrieve.
        prefix: The prefix of the config file to load.
        environment: The environment to load the config for.
        default: The default value to return if the key is not found.

    Returns:
        The nested config value.
    """
    cfg = get_config(prefix, environment)
    cur: Any = cfg
    for part in key_path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


if __name__ == "__main__":
    print("=== TRAIN-STAGING: modal_secret_name ===")
    print(
        get_config("train", "staging")["steps"]["train_pytorch_model"]["parameters"][
            "modal_secret_name"
        ]
    )

    print("\n=== COMMON modal.secret_name ===")
    print(get_config_value("modal.secret_name"))

    print("\n=== DEPLOY-PRODUCTION: << merge + interpolate >>>")
    print(get_config("deploy", "production"))

    os.environ["TEST"] = "ENV_OK"
    print("\n=== ENV interpolation:", _interpolate("${TEST}", {}), "===")
