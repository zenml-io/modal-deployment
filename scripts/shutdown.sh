#!/usr/bin/env bash

# Bash script to stop Modal apps for both sklearn and pytorch models in production and staging environments
# Suppresses Modal CLI output and error messages, only showing custom status messages

set -euo pipefail

# Colour codes
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Colour

# Define apps with their appropriate environments
APPS=(
  "pytorch-iris-production:production"
  "pytorch-iris-staging:staging"
  "sklearn-iris-production:production"
  "sklearn-iris-staging:staging"
)

for app_env in "${APPS[@]}"; do
  # Split the string by colon to get app name and environment
  IFS=':' read -r app env <<< "$app_env"
  
  echo "Stopping Modal app: $app in environment: $env"
  if modal app stop "$app" -e "$env" > /dev/null 2>&1; then
    echo -e "${GREEN}Successfully stopped '$app' in environment '$env'.${NC}"
  else
    echo -e "${RED}No app named '$app' found in environment '$env' to shutdown, or shutdown failed. Continuing...${NC}"
  fi
done
