"""Entrypoint script to train and deploy iris classification models via ZenML pipeline."""

import argparse
import logging
import os

from src.pipelines import deploy_model_pipeline, train_model_pipeline
from src.utils.model_utils import check_models_exist
from src.utils.yaml_config import get_merged_config_path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

    # Set environment variable for MODEL_STAGE to be used in constants.py
    if args.environment == "production":
        os.environ["MODEL_STAGE"] = "production"
        os.environ["MODAL_ENVIRONMENT"] = "production"
    else:
        os.environ["MODEL_STAGE"] = "staging"
        os.environ["MODAL_ENVIRONMENT"] = "staging"

    # Pre-load the configuration
    if args.train:
        # Load the train config for the environment
        config_path = get_merged_config_path("train", args.environment)
        train_model_pipeline.with_options(config_path=config_path)()

    if args.deploy:
        models_exist = check_models_exist(args.environment)

        if not models_exist:
            logger.error(
                f"❌ ERROR: Required model files not found in the '{args.environment}' Modal volume.\n"
                f"Please run the training pipeline first with:\n\n"
                f"    python run.py --train -e {args.environment}\n"
            )
            import sys

            sys.exit(1)

        logger.info(
            f"✅ Found required models in '{args.environment}' Modal volume, proceeding with deployment"
        )

        # Load the deploy config for the environment
        config_path = get_merged_config_path("deploy", args.environment)
        deploy_model_pipeline.with_options(config_path=config_path)(
            environment=args.environment
        )


if __name__ == "__main__":
    main()
