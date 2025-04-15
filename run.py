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
        "-e", "--environment",
        type=str,
        default="staging",
        choices=["staging", "production"],
        help="The Modal environment to deploy to (staging or production). When set to 'production', models are also promoted to production stage."
    )

    args = parser.parse_args()
    
    # Automatically promote to production stage if deploying to production environment
    promote_to_production = args.environment == "production"
    
    train_model_pipeline(
        deploy_models=args.deploy,
        stream_logs=args.stream_logs,
        promote_to_production=promote_to_production,
        environment=args.environment,
    )
