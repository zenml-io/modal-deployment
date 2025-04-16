"""Entrypoint script to train and deploy iris classification models via ZenML pipeline."""

import argparse
import logging

from src.pipelines import deploy_model_pipeline, train_model_pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def get_config_path(prefix: str, environment: str) -> str:
    """Get the configuration file path based on prefix and environment.

    Args:
        prefix: The configuration prefix (train or deploy)
        environment: The environment (staging or production)

    Returns:
        The path to the configuration file
    """
    return f"src/configs/{prefix}_{environment}.yaml"


def main() -> None:
    """Parse CLI args and run the train_model_pipeline with the appropriate config."""
    parser = argparse.ArgumentParser(
        description="Train and deploy iris classification models"
    )
    parser.add_argument(
        "--deploy", action="store_true", help="Deploy models to Modal after training"
    )
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument(
        "-e",
        "--environment",
        type=str,
        default="staging",
        choices=["staging", "production"],
        help="The Modal environment to deploy to (staging or production). When set to 'production', models are also promoted to production stage.",
    )
    args = parser.parse_args()

    if args.train:
        config = get_config_path("train", args.environment)
        train_model_pipeline.with_options(config_path=config)()

    if args.deploy:
        config = get_config_path("deploy", args.environment)
        deploy_model_pipeline.with_options(config_path=config)(
            environment=args.environment
        )


if __name__ == "__main__":
    main()
