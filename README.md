# DSAI3202 Assignment 2 - Machine Learning Modeling and MLOps Pipeline

This repository contains the complete MLOps lifecycle for **Assignment 2** of the Cloud Computing course. It documents the transition from raw feature engineering to a fully automated, tuned, and deployed sentiment classification model on Azure.

---

## 🚀 Mission Accomplished: The Full MLOps Lifecycle

We have successfully navigated all phases of the Machine Learning lifecycle: Training, Hyperparameter Tuning, Model Registration, Real-time Deployment, and Production Invocation.

### 1. Advanced Modeling & Automated Training
We replaced the baseline models with a **RandomForestClassifier** (wrapped in a `StandardScaler` pipeline). The training process is fully automated via **Azure DevOps CI/CD**, triggering on every push to the `assignment_2` branch.

### 2. Hyperparameter Tuning (Sweep Job)
To ensure peak performance, we executed a **Sweep Job** (`jobs/sweep_job.yml`) to explore the optimal configuration for the RandomForest model.
*   **Metric**: Maximized `test_f1_score`.
*   **Search Space**: Tuned `n_estimators` (100–300) and `max_depth` (10–20).
*   **Best Configuration**: Discovered the optimal `alpha` and `max_iter` settings, which were then promoted to the final training run.

### 3. Model Registration
The final artifact (**`model.pkl`**) has been registered in the **Azure ML Model Registry**:
*   **Name**: `amazon-review-sentiment-model`
*   **Lineage**: Fully linked to the exact training run, metrics, and dataset version.
*   **Versioning**: Azure ML now maintains a versioned history of this model for easy rollback.

### 4. Managed Online Deployment
The model is currently deployed as a **Managed Online Endpoint** for real-time inference:
*   **Endpoint Name**: `amazon-review-endpoint`
*   **Deployment Configuration**: Defines a `blue-deployment` on `Standard_F2s_v2` compute.
*   **Scoring Service**: A custom [score.py](file:///Users/mohammedmoulai/Documents/Winter%202026/cloud-computing/Assignment%202/dsai3202-lab3/src/score.py) script handles JSON data processing and prediction logic.

### 5. Production Invocation & Results
We verified the deployment using the **10% Deployment Dataset** (unseen during training/tests) through the [invoke_endpoint.py](file:///Users/mohammedmoulai/Documents/Winter%202026/cloud-computing/Assignment%202/dsai3202-lab3/src/invoke_endpoint.py) client.
*   **Success**: The endpoint successfully processes incoming feature batches and returns predictions.
*   **Insights**: Performance on the deployment split was compared with the test set to monitor for potential **data drift** in production.

---

## 🛠️ Project Structure

```
├── src/
│   ├── train.py              # Automated training logic with Pipeline
│   ├── score.py              # Magic inference script for Azure ML Endpoint
│   └── invoke_endpoint.py    # Production testing client
├── jobs/
│   ├── train_job.yml         # Standard training job definition
│   ├── sweep_job.yml         # Hyperparameter tuning configuration
│   └── deployment.yml        # Online deployment specification
├── env/
│   ├── conda.yml             # Training environment dependencies
│   └── inference_conda.yml   # Lightweight serving environment
└── azure - pipelines.yaml    # CI/CD orchestration for Azure DevOps
```

---

## 🧹 Resources & Maintenance (IMPORTANT)

> [!WARNING]
> **Active endpoints cost money.** To prevent unnecessary compute charges, always delete the endpoint after testing is complete.
> ```bash
> az ml online-endpoint delete --name amazon-review-endpoint --yes
> ```
