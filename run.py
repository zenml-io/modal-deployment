"""Entrypoint script to train and deploy iris classification models via ZenML pipeline."""

import argparse
import logging

from src.pipelines import deploy_model_pipeline, train_model_pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main() -> None:
    """Parse CLI args and run the train_model_pipeline with the appropriate config."""
    parser = argparse.ArgumentParser(
        description="Train and deploy iris classification models"
    )
    parser.add_argument(
        "--deploy", action="store_true", help="Deploy models to Modal after training"
    )
    parser.add_argument(
        "-e",
        "--environment",
        type=str,
        default="staging",
        choices=["staging", "production"],
        help="The Modal environment to deploy to (staging or production). When set to 'production', models are also promoted to production stage.",
    )

    args = parser.parse_args()

    if args.environment == "production":
        config = "src/configs/production.yaml"
    else:
        config = "src/configs/staging.yaml"

    if args.deploy:
        deploy_model_pipeline.with_options(config_path=config)(
            stream_logs=args.stream_logs,
            environment=args.environment,
        )
    else:
        train_model_pipeline.with_options(config_path=config)()


if __name__ == "__main__":
    main()
