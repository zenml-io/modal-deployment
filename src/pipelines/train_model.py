from zenml import pipeline

from src.steps.training import (
    train_pytorch_model,
    train_sklearn_model,
)


@pipeline
def train_model_pipeline() -> None:
    """Trains, registers, and deploys Iris classification models in a ZenML pipeline.

    This pipeline:
    1. Collects the active stack dependencies
    2. Trains a scikit-learn RandomForestClassifier with deployment metadata
    3. Trains a PyTorch neural network with deployment metadata
    4. Creates deployment scripts for each model using templates
    5. Optionally promotes models to production stage
    6. Optionally deploys the models to Modal using Python APIs

    Args:
        deploy_models: Whether to deploy the models to Modal
        stream_logs: Whether to stream logs from Modal deployments
        promote_to_production: Whether to promote models to production stage before deployment
        environment: The Modal environment to deploy to (staging, production, etc.)
    """
    train_sklearn_model()
    train_pytorch_model()
