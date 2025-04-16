# This is the secret name used in Modal to store details for ZenML to load
# properly
MODAL_SECRET_NAME = "modal-deployment-credentials"

# Model configuration - will be obtained from ZenML
MODEL_NAME = "iris-classification"
MODEL_STAGE = "latest"  # Default to latest version, will be updated by CLI args

# Generate a deployment ID using model stage instead of random UUID
SKLEARN_DEPLOYMENT_ID = f"sklearn-iris-{MODEL_STAGE}"
PYTORCH_DEPLOYMENT_ID = f"pytorch-iris-{MODEL_STAGE}"

# Map prediction indices to species names - this is a true constant and should live here.
SPECIES_MAPPING = {0: "setosa", 1: "versicolor", 2: "virginica"}
