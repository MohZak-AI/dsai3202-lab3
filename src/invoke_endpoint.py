import requests
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Endpoint Details - Update these with your values
# --------------------------------------------------
ENDPOINT_URL = "<YOUR_ENDPOINT_SCORING_URL>"
API_KEY = "<YOUR_ENDPOINT_KEY>"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def main():
    """
    Simulates production traffic by loading the 10% deployment dataset
    and sending samples to the managed online endpoint for prediction.
    """
    print("--- Starting Endpoint Invocation Test ---")
    
    # 1. Load the deployment dataset
    # In a real environment, you would use mltable or load directly from ADLS
    # For this simulation, we assume the dataset is available for testing
    try:
        # Placeholder path - in a live script, we'd use the registered Data Asset
        # df = pd.read_parquet("path/to/deployment/data")
        print("1. Loading deployment dataset (10% partition)...")
        # --- Mocking for example purposes ---
        # df = ... 
        # X = df.drop(columns=['label', 'asin', 'reviewerID', ...])
        # y_true = df['label']
        print("   Status: Dataset load logic configured.")
    except Exception as e:
        print(f"   Error: Load logic failed: {str(e)}")
        return

    # 2. Build the Payload (Example using 5 sample rows)
    print("2. Constructing inference payload for sample batch...")
    # This matches the expected [{"data": [[...], [...]...]}] format in score.py
    # payload = {"data": X.iloc[:5].values.tolist()}
    
    # 3. Send POST Request to Endpoint
    print(f"3. Sending POST request to: {ENDPOINT_URL}")
    try:
        # Example dummy request to illustrate structure (replace with real data)
        dummy_data = [[0.1] * 900] # Representative of our feature length
        payload = {"data": dummy_data}
        
        response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            result = response.json()
            predictions = result.get("predictions")
            print("   Success! Predictions received from model.")
            print(f"   Sample Predictions: {predictions}")
            
            # 4. Compute Metrics (Simulation)
            # accuracy = accuracy_score(y_true[:5], predictions)
            # print(f"4. Deployment Accuracy (Batch): {accuracy:.4f}")
        else:
            print(f"   Error: Endpoint returned {response.status_code} - {response.text}")

    except Exception as e:
        print(f"   Error: Failed to connect to endpoint: {str(e)}")

    print("--- Test Complete ---")

if __name__ == "__main__":
    main()
