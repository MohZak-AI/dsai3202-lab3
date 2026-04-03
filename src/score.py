import json
import os
import joblib
import pandas as pd
import logging

# Set up logging for easier debugging in Azure ML Studio
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None

def init():
    """
    Called when the container is initialized.
    Loads the model artifact from the Azure ML model directory.
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created by Azure ML
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pkl")
    logger.info(f"Loading model from: {model_path}")
    
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def run(raw_data):
    """
    Called for every request to the endpoint.
    Processes the raw JSON input and returns predictions.
    """
    logger.info("Received request for inference.")
    try:
        # Load the input JSON
        # Standard Azure ML request format is {"data": [[...], [...]]}
        request_data = json.loads(raw_data)
        data = request_data.get("data")
        
        if data is None:
            return {"error": "No 'data' key found in JSON body."}

        # The model is a scikit-learn Pipeline that expects a DataFrame or array
        # Our training used a consistent feature set, so we convert input to DataFrame
        X = pd.DataFrame(data)
        
        # Get predictions (0 or 1)
        preds = model.predict(X)
        
        # Return as list for JSON serialization
        return {"predictions": preds.tolist()}
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        return {"error": str(e)}
