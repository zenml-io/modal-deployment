from zenml import step, pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from zenml.client import Client
from rich import print
from typing import Annotated


@step
def train_sklearn_model() -> Annotated[RandomForestClassifier, "sklearn_model"]:
    # Load dataset
    X, y = load_iris(return_X_y=True)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model


@pipeline(enable_cache=False)
def train_model_pipeline():
    model = train_sklearn_model()


if __name__ == "__main__":
    train_model_pipeline()

    zenml_client = Client()

    sklearn_model = zenml_client.get_artifact_version(
        name_id_or_prefix="sklearn_model"
    ).load()

    print(type(sklearn_model))
