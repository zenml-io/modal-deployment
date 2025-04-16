from zenml import pipeline


from src.steps.deployment import modal_deployment


@pipeline
def deploy_model_pipeline(
    stream_logs: bool = False,
    promote_to_production: bool = False,
    environment: str = "staging",
):
    modal_deployment(
        stream_logs=stream_logs,
        environment_name=environment,
    )
