import argparse
import os
import time
import azureml.mlflow
import mlflow
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Arguments
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()

# --------------------------------------------------
# Load data
# --------------------------------------------------
def load_data(path):
    # Azure ML passes directories for uri_folder; we need to find the parquet files inside
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    files = [f for f in os.listdir(path) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {path}")
        
    return pd.concat([pd.read_parquet(os.path.join(path, f)) for f in files], ignore_index=True)

# --------------------------------------------------
# Labels
# --------------------------------------------------
def create_labels(df):
    if "overall" not in df.columns:
        raise RuntimeError("Column 'overall' is missing. You had one job.")
    # Binary classification: 4-5 stars are positive (1), others are negative (0)
    df["label"] = (df["overall"] >= 4).astype(int)
    return df

# --------------------------------------------------
# Features
# --------------------------------------------------
def get_feature_cols(df):
    """
    Identifies the engineered feature columns from Lab 4.
    """
    feature_cols = []
    
    # 1. Length features
    feature_cols.extend(["review_length_words", "review_length_chars"])
    
    # 2. Sentiment features
    feature_cols.extend(["sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_compound"])
    
    # 3. TF-IDF features
    tfidf_cols = sorted([c for c in df.columns if c.startswith("tfidf_")])
    feature_cols.extend(tfidf_cols)
    
    # 4. SBERT features
    bert_cols = sorted([c for c in df.columns if c.startswith("bert_embedding_")])
    feature_cols.extend(bert_cols)
    
    return feature_cols

def build_feature_matrix(df, feature_cols):
    """
    Constructs the feature matrix using the provided column list.
    """
    # Check for missing columns in this specific split
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"Warning: {len(missing_cols)} feature columns are missing from this split.")
        # Fill missing columns with 0 to maintain shape
        for col in missing_cols:
            df[col] = 0.0

    X = df[feature_cols].values
    
    if X.shape[1] == 0:
        raise RuntimeError("Feature matrix is empty. Did your pipeline even run?")
        
    return X

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
def evaluate(model, X, y, split):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    mlflow.log_metric(f"{split}_accuracy", acc)
    print(f"{split} accuracy: {acc:.4f}")
    return acc

def main():
    args = parse_args()
    start_time = time.time()
    
    # Initialize MLflow
    mlflow.start_run()
    
    print("Loading datasets produced in Lab 4...")
    train_df = load_data(args.train_data)
    val_df = load_data(args.val_data)
    test_df = load_data(args.test_data)
    
    print("Converting ratings into binary classification labels...")
    train_df = create_labels(train_df)
    val_df = create_labels(val_df)
    test_df = create_labels(test_df)
    
    print("Constructing the feature matrix...")
    # Determine feature columns from training set ONLY
    feature_cols = get_feature_cols(train_df)
    print(f"Identified {len(feature_cols)} features.")

    X_train = build_feature_matrix(train_df, feature_cols)
    y_train = train_df["label"]
    
    X_val = build_feature_matrix(val_df, feature_cols)
    y_val = val_df["label"]
    
    X_test = build_feature_matrix(test_df, feature_cols)
    y_test = test_df["label"]
    
    print(f"Training on {X_train.shape[0]} samples...")
    
    # Training
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    print("Evaluating model performance...")
    evaluate(model, X_train, y_train, "train")
    evaluate(model, X_val, y_val, "val")
    evaluate(model, X_test, y_test, "test")
    
    print("Saving model artifact...")
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "model.pkl")
    joblib.dump(model, model_path)
    
    # Log model to MLflow
    mlflow.sklearn.log_model(model, "logistic_regression_model")
    
    runtime = time.time() - start_time
    mlflow.log_metric("training_runtime_seconds", runtime)
    print(f"Done. Total runtime: {runtime:.2f}s")
    
    mlflow.end_run()

if __name__ == "__main__":
    main()