# Customer Churn Prediction Model

## Overview
This project builds a machine learning model to predict customer churn — identifying customers likely to stop using a service. Early identification helps businesses improve retention strategies and reduce revenue loss.

## Data
The dataset contains customer information such as demographics, account details, and service usage. Important columns include:
- Customer demographics (Age, Gender, Geography)
- Account info (Tenure, Balance, Number of products)
- Activity indicators (Credit card, Active membership)
- Target: `Exited` (1 if customer churned, 0 otherwise)

## Features and Preprocessing
- Handled missing values and encoded categorical features.
- Scaled numerical features using standardization.
- Built a preprocessing pipeline to ensure consistent feature transformation for training and new data.

## Models Compared
- Dummy Classifier: Baseline model making trivial predictions.
- Logistic Regression: Linear model to estimate churn probability.
- Random Forest: Ensemble model capturing nonlinear feature interactions.

## Model Training and Evaluation
- Data split into training and test sets.
- Models trained on training data.
- Performance evaluated on test data using metrics:
  - Precision-Recall AUC (PR-AUC)
  - ROC AUC
  - F1 score (optimized by threshold tuning)
  
## Threshold Tuning
- Search for best probability cutoff that maximizes F1 score for Logistic Regression.
- Enables better balance between precision and recall in predictions.

## Results
- Random Forest achieved the best PR-AUC and ROC-AUC.
- Final model saved using `joblib` for future deployment.

## Usage
- Load the saved model pipeline.
- Prepare new data with the same features.
- Use `predict()` or `predict_proba()` to generate churn predictions.
  
## Dependencies
- Python 3.x
- scikit-learn
- pandas
- numpy
- joblib

## How to Run
1. Train model: Run `train.py` to fit models and save the best one.
2. Predict churn: Use `predict.py` with new customer data.
3. Evaluate performance with provided scripts/notebooks.

## Contact
For questions or collaboration, reach out at [email@example.com].

---

This README covers project objectives, data, modeling steps, evaluation, and usage to guide users and maintainers.
