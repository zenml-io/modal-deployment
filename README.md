# Model Deployment with Modal and ZenML

This project demonstrates how to deploy a machine learning model trained with ZenML using Modal for serverless deployment.

## Manual Deployment Process

This section shows how to manually deploy a model using Modal without using
ZenML explicitly for deployment to Modal.

### Step 1: Train and Register Your Model

First, run the training pipeline to train and register a model in ZenML:

```bash
python run.py
```

This script:
- Loads the Iris dataset
- Trains a RandomForestClassifier on the data
- Registers the model as a ZenML artifact named "sklearn_model"

### Step 2: Deploy the Model with Modal

Modal offers two ways to run your service:

#### Development Mode

For testing and development, use:

```bash
modal serve modal_deployer.py
```

This will start a development server with hot-reloading.

#### Production Deployment

For a permanent, production deployment:

```bash
modal deploy modal_deployer.py
```

This will deploy your service to Modal's infrastructure with a persistent endpoint.

### How It Works

#### `modal_deployer.py` Explained

The deployment script (`modal_deployer.py`):

1. Creates a Modal image with all necessary dependencies
   - Installs Python 3.12.3
   - Extracts requirements from your pyproject.toml
   - Adds integration requirements for ZenML, AWS, S3, scikit-learn, etc.
   - Uses Modal secrets to authenticate with ZenML

2. Defines a `ModelDeployer` class that:
   - Loads the model from ZenML storage
   - Provides prediction endpoints

3. Sets up a FastAPI app with:
   - Health check endpoints
   - Prediction endpoint that receives feature data and returns predictions

### Using the Deployed Model

Once deployed, you can query your model using HTTP requests:

```bash
curl -X POST \
  https://your-app-name--model-api.modal.run/predict/sklearn \
  -H 'Content-Type: application/json' \
  -d '{"features": [[5.1, 3.5, 1.4, 0.2], [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9]]}'
```

The response will include predictions and probability scores (if available):

```json
{
  "predictions": [0, 2, 2],
  "probabilities": [[0.97, 0.02, 0.01], [0.0, 0.05, 0.95], [0.0, 0.14, 0.86]]
}
```

## ZenML-Managed Deployment

This is an alternative version which:

- adds an extra step to run.py which takes the model artifact, saves it to a
  local directory and then builds an image
