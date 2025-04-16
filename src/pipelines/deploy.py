from zenml import pipeline

from src.steps.deployment import modal_deployment


@pipeline
def deploy_model_pipeline(environment: str = "staging"):
    modal_deployment(environment_name=environment)
