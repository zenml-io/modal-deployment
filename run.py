import argparse

from src.pipelines import train_model_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and deploy iris classification models"
    )
    parser.add_argument(
        "--deploy", action="store_true", help="Deploy models to Modal after training"
    )
    parser.add_argument(
        "--stream-logs", action="store_true", help="Stream logs from Modal deployments"
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Promote models to production stage before deployment",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="staging",
        choices=["staging", "production"],
        help="The Modal environment to deploy to (staging or production)"
    )

    args = parser.parse_args()

    # If using --production flag for deployment, automatically set environment to production
    environment = "production" if args.production else args.environment
    
    train_model_pipeline(
        deploy_models=args.deploy,
        stream_logs=args.stream_logs,
        promote_to_production=args.production,
        environment=environment,
    )
