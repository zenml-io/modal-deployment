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

## Project Structure

- `run.py`: Entry point for the training and deployment pipeline
- `src/`: Core source code
  - `configs/`: Environment-specific configuration files (staging/production)
  - `pipelines/`: ZenML pipeline definitions
  - `steps/`: ZenML step implementations for training and deployment
  - `templates/`: Modal deployment templates for different model types
- `scripts/`: Utility scripts
  - `format.sh`: Code formatting script
  - `shutdown.sh`: Script to stop deployments in staging and production
- `design/`: Design documents and architecture diagrams

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

Here are sample curl commands to interact with the deployed endpoints:

### Health Check
```bash
curl -X GET https://<your-modal-deployment-url>/health
```

### Make Predictions (for either sklearn or pytorch deployment)
```bash
curl -X POST https://<your-modal-deployment-url>/predict \
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
