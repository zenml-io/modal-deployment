from zenml import pipeline
from zenml.enums import ModelStages

from src.steps.deployment import modal_deployment
from src.steps.training import (
    get_stack_dependencies,
    train_pytorch_model,
    train_sklearn_model,
)


@pipeline(enable_cache=False, name="iris_model_training_and_deployment")
def train_model_pipeline(
    deploy_models: bool = False,
    stream_logs: bool = False,
    promote_to_production: bool = False,
    environment: str = "staging",
):
    """ZenML pipeline that trains, registers, and deploys Iris classification models.

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
    stack_dependencies = get_stack_dependencies()
    train_sklearn_model(stack_dependencies=stack_dependencies)
    train_pytorch_model(stack_dependencies=stack_dependencies)

    # Determine stage for promotion if we're deploying to production
    promote_to_stage = ModelStages.PRODUCTION if promote_to_production else None

    modal_deployment(
        deploy=deploy_models,
        stream_logs=stream_logs,
        promote_to_stage=promote_to_stage,
        environment_name=environment,
        after=["train_sklearn_model", "train_pytorch_model"],
    )
