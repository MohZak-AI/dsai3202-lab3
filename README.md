# DSAI3202 Assignment 2 - Machine Learning Modeling and MLOps Pipeline

This repository contains the end-to-end Machine Learning pipeline for **Assignment 2** of the Cloud Computing course. The goal of this assignment is to train a classification model on Amazon Electronics reviews, optimize its performance, and automate the lifecycle using MLOps best practices on Azure.

## Project Scope

Building on the feature extraction work from previous labs (length, sentiment, TF-IDF, and SBERT embeddings), this assignment focuses on the **modeling**, **evaluation**, and **CI/CD** phases.

### 1. Machine Learning Model
We implemented a robust classification model to predict whether a review is positive (4-5 stars) or negative (1-3 stars).
*   **Algorithm**: **RandomForestClassifier** (200 estimators, max depth 15, balanced class weights). This was chosen over simple Logistic Regression to better handle non-linear patterns in high-dimensional text features.
*   **Pipeline Strategy**: A scikit-learn `Pipeline` integrates a `StandardScaler` with the model. This ensures features are properly normalized automatically during both training and future batch/real-time inference.
*   **Data Handling**: Uses a custom loading script designed to handle multi-part Parquet files produced by the Lab 4 feature engineering pipeline.

### 2. MLOps & Tracking
We integrated **MLflow** with Azure ML to track every run with a comprehensive suite of metrics:
*   **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
*   **ROC-AUC** (Area Under Curve): Calculated using prediction probabilities to give a better view of classifier quality than accuracy alone.
*   **Training Runtime**: Logged for performance auditing.
*   **Model Artifacts**: The trained model is saved as **`model.pkl`** in the Azure ML `outputs` directory for easy deployment.

### 3. CI/CD Integration
The training workflow is automated using **Azure DevOps Pipelines**:
*   **Trigger**: Automatic job submission on every push to the **`assignment_2`** branch.
*   **Pipeline Configuration**: `azure - pipelines.yaml` handles the authentication via the `SC-UDST-CCIT-DSAI3202-2` service connection.
*   **Azure ML Job**: The job is defined in `jobs/train_job.yml` and executes on the `lab4-60301919` compute cluster using the `env/conda.yml` environment.

## Getting Started

### Prerequisites
*   Azure CLI with the `ml` extension installed.
*   Access to the Azure ML Workspace: `Amazon-Electronics-Lab-60301919`.
*   Active service connection in Azure DevOps: `SC-UDST-CCIT-DSAI3202-2`.

### Manual Submission
To submit the training job manually via the Azure CLI:
```bash
az ml job create --file jobs/train_job.yml --resource-group rg-60301919 --workspace-name Amazon-Electronics-Lab-60301919
```

### Automated Submission
Simply commit and push your changes to the `assignment_2` branch:
```bash
git add .
git commit -m "Trigger training job"
git push origin assignment_2
```
Azure DevOps will automatically trigger the pipeline defined in `azure - pipelines.yaml`.

## Results
Once the job is completed in Azure ML Studio:
1.  Navigate to the **"Metrics"** tab to view the performance (AUC, F1, etc.).
2.  Navigate to the **"Outputs + logs"** tab to download the final **`model.pkl`** artifact.