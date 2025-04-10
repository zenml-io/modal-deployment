from zenml import step, pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


@step
def train_sklearn_model() -> RandomForestClassifier:
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


@pipeline
def train_model_pipeline():
    model = train_sklearn_model()


if __name__ == "__main__":
    train_model_pipeline()
