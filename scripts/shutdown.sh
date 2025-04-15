#!/usr/bin/env bash

# Bash script to stop Modal apps for both sklearn and pytorch models in production and staging environments
# Suppresses Modal CLI output and error messages, only showing custom status messages

set -euo pipefail

# Colour codes
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Colour

APPS=("iris-model-sklearn" "iris-model-pytorch")
ENVS=("production" "staging")

for app in "${APPS[@]}"; do
  for env in "${ENVS[@]}"; do
    echo "Stopping Modal app: $app in environment: $env"
    if modal app stop "$app" -e "$env" > /dev/null 2>&1; then
      echo -e "${GREEN}Successfully stopped '$app' in environment '$env'.${NC}"
    else
      echo -e "${RED}No app named '$app' found in environment '$env' to shutdown, or shutdown failed. Continuing...${NC}"
    fi
  done
done
