# Modal-ZenML Model Deployment

A framework for training machine learning models with ZenML and deploying them to [Modal](https://modal.com).

## Overview

This project demonstrates an end-to-end ML workflow:

1. Training ML models (scikit-learn and PyTorch)
2. Registering them with ZenML's Model Control Plane
3. Deploying them to Modal for scalable, serverless inference

## Prerequisites

- Python 3.12+ (recommended)
- Modal account and CLI setup
- ZenML server (if using remote registry)
- Docker (for local development)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd modal-deployment
```

2. Install dependencies:

```bash
# assuming you have uv installed
uv pip install -r pyproject.toml
```

3. Set up Modal CLI:

```bash
modal token new
```

4. Set up Modal environments:

```bash
modal environment create staging
modal environment create production
```

5. Set up Modal secrets for ZenML access:

```bash
# Set your ZenML server details as variables
ZENML_URL="<your-zenml-server-url>"
ZENML_API_KEY="<your-zenml-api-key>"

# Create secrets for staging environment
modal secret create modal-deployment-credentials \
   ZENML_STORE_URL=$ZENML_URL \
   ZENML_STORE_API_KEY=$ZENML_API_KEY \
   -e staging

# Create secrets for production environment
modal secret create modal-deployment-credentials \
   ZENML_STORE_URL=$ZENML_URL \
   ZENML_STORE_API_KEY=$ZENML_API_KEY \
   -e production
```

6. Set up Modal volumes:

```bash
# Create staging volume
modal volume create iris-staging-models -e staging

# Create production volume
modal volume create iris-prod-models -e production
```

## Project Structure

- `run.py`: Entry point for the training and deployment pipeline
- `app/`: Modal deployment application code
  - `deployment_template.py`: The main Modal app implementation with FastAPI integration
  - `schemas.py`: Iris model, prediction, and API endpoint schemas for Modal deployment
- `src/`: Core source code
  - `configs/`: Environment-specific configuration files (staging/production)
  - `pipelines/`: ZenML pipeline definitions
  - `steps/`: ZenML step implementations for training and deployment
  - `schemas/`: Iris model and prediction schemas for training
- `scripts/`: Utility scripts
  - `format.sh`: Code formatting script
  - `shutdown.sh`: Script to stop deployments in staging and production

## Configuration with YAML Anchors

This project uses YAML anchor keys for efficient configuration management across environments:

```yaml
# In common.yaml, we define shared configuration with an anchor
&COMMON
modal:
  secret_name: "modal-deployment-credentials"
# ... more common configuration

# In environment-specific configs, we merge the common config
<<: *COMMON
# ... environment-specific overrides and additions
```

The `&COMMON` anchor in `common.yaml` defines shared settings, while `<<: *COMMON` in other config files merges these settings before adding environment-specific configurations. This approach maintains consistent base settings while allowing per-environment customization of parameters like volume names and deployment stages.

## Usage

### Training and Deployment

To run the pipeline for training models and/or deploying them:

```bash
# Train models only
python run.py --train

# Deploy to Modal (staging environment by default)
python run.py --deploy

# Train models and deploy to Modal (staging environment)
python run.py --train --deploy

# Deploy to production environment
python run.py --deploy -e production

# Train and deploy to production environment
python run.py --train --deploy -e production
```

## API Endpoints

Once deployed, each model service (sklearn or pytorch) exposes the following endpoints:

- `GET /`: Welcome message with deployment/model info
- `GET /health`: Health check endpoint
- `GET /url`: Returns the deployment URL
- `POST /predict`: Make predictions using the model

#### Example prediction request (`/predict`):

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

The response includes the predicted class index and probabilities:

```json
{
  "prediction": 0,
  "prediction_probabilities": [0.97, 0.02, 0.01],
  "species_name": "setosa"
}
```

## Sample API Requests

Here are sample curl commands to interact with the deployed endpoints. You can
find the URL of your deployment in the Modal dashboard as well as in the ZenML
dashboard. It will have been output to the terminal when you deployed the model.
(Note that there are two URLs, one for the PyTorch deployment and one for the
`scikit-learn` deployment.)

First, set your deployment URL as a variable (including the https:// prefix):

```bash
export MODAL_URL="https://your-modal-deployment-url"  # Replace with your actual URL
# For example: export MODAL_URL="https://someuser-staging--pytorch-iris-predictor-staging.modal.run"
```

### Health Check

```bash
curl -X GET $MODAL_URL/health
```

### Make Predictions (for either sklearn or pytorch deployment)

```bash
curl -X POST $MODAL_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

Response:

```json
{
  "prediction": 0,
  "prediction_probabilities": [0.97, 0.02, 0.01],
  "species_name": "setosa"
}
```

## Advanced Features

### Configuration Files

The project uses environment-specific configuration files located in `src/configs/`:

- `train_staging.yaml`: Configuration for training in staging environment
- `train_production.yaml`: Configuration for training in production environment
- `deploy_staging.yaml`: Configuration for deployment in staging environment
- `deploy_production.yaml`: Configuration for deployment in production environment

These configuration files control various aspects of the pipelines and deployments. You can modify these files to customize behavior without changing code.

### Model Stages

The system integrates with ZenML's model registry and supports environment-specific deployments to Modal.

### Integration with Modal

The deployment uses Modal's features like:

- Secret management for ZenML credentials
- Python package caching for fast deployments
- Serverless scaling based on demand
- Volume mount for model storage

### Stopping Deployments

To stop all deployments in both staging and production environments:

```bash
./scripts/shutdown.sh
```

This is particularly useful during development or when you need to clean up resources.

## Troubleshooting

- **Missing ZenML credentials**: Ensure Modal secret is correctly set up
- **Model loading errors**: Check ZenML model registry or `/health` endpoint
- **Deployment failures**: Check logs in the Modal dashboard
- **Invalid function call error**: Ensure you're using the correct URL format for your deployment
