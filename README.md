# ZenML Docker Deployment for ML Models

This project demonstrates how to package trained scikit-learn and PyTorch models into a Docker image for deployment using ZenML pipelines.

## Features

- Trains both scikit-learn RandomForestClassifier and PyTorch neural network models on the Iris dataset
- Registers models with ZenML's model registry
- Builds a Docker image that dynamically loads models from ZenML at runtime
- Exposes a FastAPI service with endpoints for health checks and predictions
- Supports model-specific endpoints and a generic prediction endpoint

## End-to-End Workflow

This project implements a complete ML workflow:

1. **Model Training**: The script trains both sklearn and PyTorch models on the Iris dataset
2. **Model Registry**: Models are registered in the ZenML model registry with proper versioning
3. **Docker Packaging**: An API service is packaged into a Docker container
4. **Dynamic Model Loading**: The service loads the latest production model versions at runtime
5. **API Deployment**: FastAPI provides prediction endpoints for both models

## Requirements

- Python 3.9+
- Docker
- ZenML server (with API key)
- Dependencies in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install dependencies
uv add zenml scikit-learn torch fastapi docker
```

## Usage

### Running the Pipeline

The pipeline trains models, registers them with ZenML, and builds a Docker image:

```bash
# Set ZenML environment variables if not already set
export ZENML_STORE_URL="your-zenml-server-url"
export ZENML_STORE_API_KEY="your-zenml-api-key"

# Run the pipeline
uv run python zenml_e2e_modal_deployment.py
```

This will:
1. Train a scikit-learn model and a PyTorch model
2. Register the models with ZenML
3. Build a Docker image for the model server
4. Print the image name/tag
5. Automatically deploy the container if ZenML environment variables are set

### Running the Docker Container

The pipeline will automatically try to run the container if environment variables are set. You can also run it manually:

```bash
docker run -d \
  -p 8000:8000 \
  -e ZENML_STORE_URL="your-zenml-server-url" \
  -e ZENML_STORE_API_KEY="your-zenml-api-key" \
  -e SKLEARN_MODEL_NAME="sklearn_model" \
  -e SKLEARN_MODEL_STAGE="production" \
  -e PYTORCH_MODEL_NAME="pytorch_model" \
  -e PYTORCH_MODEL_STAGE="production" \
  --name iris-predictor \
  <image-name>
```

Replace `<image-name>` with the image name printed after running the pipeline.

### Testing the API

Once the container is running, you can test the API with:

```bash
# Check health status
curl http://localhost:8000/health

# View debug info (includes model loading errors)
curl http://localhost:8000/debug

# Test sklearn model
curl -X POST http://localhost:8000/predict/sklearn \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# Test PyTorch model
curl -X POST http://localhost:8000/predict/pytorch \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

### API Endpoints

The deployed model service exposes the following endpoints:

- `GET /health` - Health check endpoint, shows which models are loaded
- `GET /debug` - Debug endpoint with loading errors and environment info
- `POST /predict/sklearn` - Make predictions using the scikit-learn model
- `POST /predict/pytorch` - Make predictions using the PyTorch model
- `POST /predict?model_type=<type>` - Make predictions using specified model (default: sklearn)

## Project Structure

- `zenml_e2e_modal_deployment.py` - Main ZenML pipeline script that:
  - Defines model training steps
  - Registers models with ZenML
  - Builds and deploys the Docker image
- `app/` - Docker application files
  - `main.py` - FastAPI application that loads models from ZenML
  - `Dockerfile` - Docker image definition
  - `requirements.txt` - Python dependencies for the Docker image

## Important Implementation Details

### PyTorch Model Loading

The application includes special handling for PyTorch model loading:

1. The IrisModel class definition is included directly in main.py to ensure the model architecture is available
2. Models are loaded using a fallback strategy:
   - First attempts standard loading
   - Falls back to creating a new model with the same architecture if needed

### ZenML Model Registry

Models are stored in ZenML with:
- Named versions (e.g., "production")
- Associated metadata for tracking training metrics
- Artifact references that make them accessible via the ZenML API

### Docker Environment Considerations

The Docker container:
- Runs as a non-root user for security
- Sets HOME=/home/app to prevent permission issues
- Requires ZenML credentials to access models at runtime

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   - Check the `/debug` endpoint to see specific loading errors
   - Ensure ZenML environment variables are correctly set
   - Verify model names and stages match what's in the ZenML registry

2. **PyTorch Serialization Issues**
   - PyTorch models can have compatibility issues between environments
   - The app tries multiple loading strategies to handle this
   - Consider using ONNX format for more robust deployments

3. **Permission Errors**
   - If you see "Permission denied: '/nonexistent'" errors, ensure the HOME environment variable is set in the Dockerfile

4. **ZenML API Compatibility**
   - The ZenML client API may change between versions
   - Check your ZenML server and client versions match

## Extending

To extend this project:

1. Add new model types by creating additional training steps
2. Register the new models with ZenML
3. Update environment variables in the container to load the new models
4. Extend the FastAPI application with new endpoints for the new models

## Future Improvements

- Add model versioning controls via the API
- Implement A/B testing between model versions
- Add automatic monitoring of model performance
- Support ONNX model format for better portability

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
