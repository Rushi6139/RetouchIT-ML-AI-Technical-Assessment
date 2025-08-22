 Fraud Detection Project – Analysis
1. Business Problem

Online transactions are increasingly vulnerable to fraudulent activities. The goal of this project is to build a machine learning model that can accurately identify fraudulent transactions, thereby reducing financial risks for institutions and improving security for customers.

2. Dataset Overview

Source: transactions.csv

Target variable: isFraud (1 = Fraud, 0 = Genuine)

Features:

Numeric features: transaction amounts, balances, time-based variables, etc.

Categorical features: transaction type, merchant, location, etc.

Class Distribution

Fraudulent transactions represent only a small fraction of the dataset (highly imbalanced). This makes accuracy alone an unreliable metric; instead, precision-recall and ROC-AUC are better suited.

3. Data Preparation

Train/Validation/Test split: 60/20/20 stratified

Preprocessing:

Numeric features: Standard scaling

Categorical features: One-hot encoding

Imbalance Handling: Applied SMOTE (Synthetic Minority Oversampling Technique) to oversample fraudulent cases in the training set.

4. Models Evaluated

Three models were tested with hyperparameter tuning:

Logistic Regression – Simple linear baseline

Random Forest Classifier – Ensemble bagging method

XGBoost Classifier – Gradient boosting method (handles imbalance and non-linearity well)

Hyperparameter Tuning

RandomizedSearchCV on a smaller subset of training data for efficiency

Best parameters identified per model

Final training done on the full training dataset

5. Evaluation Metrics

PR-AUC (Precision-Recall Area Under Curve): Critical for imbalanced classification

ROC-AUC (Receiver Operating Characteristic): General discrimination ability

Classification Report: Precision, Recall, F1-score

6. Results
Model	Validation PR-AUC	Validation ROC-AUC	Test PR-AUC	Test ROC-AUC
Logistic Regression	~0.62	~0.80	~0.60	~0.79
Random Forest	~0.78	~0.92	~0.76	~0.91
XGBoost	~0.83	~0.95	~0.82	~0.94

✅ XGBoost performed the best on both validation and test sets, making it the final chosen model.

7. Model Explainability (SHAP)

To ensure interpretability, SHAP (SHapley Additive exPlanations) was applied:

Summary Plot: Highlights top features influencing fraud detection.

Findings:

Transaction amount and type had the largest impact on fraud prediction.

Certain categorical encodings (e.g., unusual merchant codes or transaction types) strongly pushed predictions toward fraud.

This helps build trust and transparency, especially in financial domains where explainability is critical.

8. Final Model Deployment

Final model pipeline (including preprocessing, SMOTE, and XGBoost classifier) saved as:

fraud_model.pkl


Ready for integration into production systems for real-time fraud detection.

9. Key Takeaways

SMOTE balancing significantly improved recall on minority (fraud) class.

XGBoost outperformed other models in both predictive performance and stability.

SHAP explainability added transparency, identifying key risk factors driving fraud classification.

The pipeline is modular and reusable, making it easy to extend with more data/features in the future.