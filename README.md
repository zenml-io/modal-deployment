# Modal-ZenML Model Deployment

A framework for training machine learning models with ZenML and deploying them to Modal's serverless platform.

## Overview

This project demonstrates an end-to-end ML workflow:

1. Training ML models (scikit-learn and PyTorch)
2. Registering them with ZenML's model registry
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

- `zenml_e2e_modal_deployment.py`: Full pipeline for training and deploying both scikit-learn and PyTorch models
- `templates/`: Deployment templates for different model types
- `design/`: Design documents and architecture diagrams

## Usage

### Full Pipeline with Deployment

To run the complete pipeline that trains both scikit-learn and PyTorch models and optionally deploys them:

```bash
# Train models only
python zenml_e2e_modal_deployment.py

# Train models and deploy to Modal
python zenml_e2e_modal_deployment.py --deploy

# Train models, promote to production, and deploy to Modal with logs
python zenml_e2e_modal_deployment.py --deploy --production --stream-logs
```

## API Endpoints

Once deployed, the model service exposes the following endpoints:

- `GET /`: Welcome message
- `GET /health`: Health check endpoint
- `POST /predict/sklearn`: Make predictions using the scikit-learn model
  ```json
  {
    "features": [[5.1, 3.5, 1.4, 0.2]]
  }
  ```

The response includes predictions and probabilities:
```json
{
  "predictions": [0],
  "probabilities": [[0.97, 0.02, 0.01]]
}
```

## Sample API Requests

Here are sample curl commands to interact with the deployed endpoints:

### Health Check
```bash
curl -X GET https://<your-modal-deployment-url>/health
```

### Make Predictions with scikit-learn Model
```bash
curl -X POST https://<your-modal-deployment-url>/predict/sklearn \
  -H "Content-Type: application/json" \
  -d '{"features": [[5.1, 3.5, 1.4, 0.2]]}'
```

Response:
```json
{
  "predictions": [0],
  "probabilities": [[0.97, 0.02, 0.01]]
}
```

## Advanced Features

### Model Stages

The system supports ZenML model stages like "production", "staging", and "latest".

To promote a model to production before deployment:

```bash
python zenml_e2e_modal_deployment.py --deploy --production
```

### Integration with Modal

The deployment uses Modal's features like:
- Secret management for ZenML credentials
- Python package caching for fast deployments
- Serverless scaling based on demand

## Troubleshooting

- **Missing ZenML credentials**: Ensure Modal secret is correctly set up
- **Model loading errors**: Check ZenML model registry or `/health` endpoint
- **Deployment failures**: Use `--stream-logs` for detailed Modal logs
