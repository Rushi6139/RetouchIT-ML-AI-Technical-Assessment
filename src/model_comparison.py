# src/model_comparison.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from src.preprocessing import get_preprocessor
import numpy as np

# Load processed data
data_path = 'data/processed/transactions_clean.csv'
df = pd.read_csv(data_path)

# Features and target
X = df.drop(columns=['isFraud'])
y = df['isFraud']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing
preprocessor = get_preprocessor()
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Handle class imbalance with SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_transformed, y_train)

# Define models and hyperparameter grids
models = {
    "log_reg": LogisticRegression(class_weight='balanced', max_iter=1000),
    "rf": RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42),
    "xgb": XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=42)
}

param_grids = {
    "log_reg": {"C": np.logspace(-3, 3, 7), "penalty": ['l1', 'l2'], "solver": ['liblinear']},
    "rf": {"n_estimators": [100, 200, 500], "max_depth": [5, 10, None], "min_samples_split": [2, 5, 10]},
    "xgb": {"n_estimators": [100, 200], "max_depth": [3, 6, 10], "learning_rate": [0.01, 0.1, 0.2]}
}

best_estimators = {}

# Training with RandomizedSearchCV
for name, model in models.items():
    print(f"Training {name}...")
    search = RandomizedSearchCV(
        model,
        param_distributions=param_grids[name],
        n_iter=10,
        scoring='average_precision',
        cv=3,
        verbose=1,
        n_jobs=1,
        random_state=42
    )
    search.fit(X_train_res, y_train_res)
    best_estimators[name] = search.best_estimator_

    # Evaluate on test set
    y_scores = best_estimators[name].predict_proba(X_test_transformed)[:, 1]
    pr_auc = average_precision_score(y_test, y_scores)
    print(f"{name} Precision-Recall AUC: {pr_auc:.4f}\n")

# Choose best model (highest PR-AUC)
best_model_name = max(best_estimators, key=lambda k: average_precision_score(
    y_test, best_estimators[k].predict_proba(X_test_transformed)[:, 1]
))
best_model = best_estimators[best_model_name]
print(f"Selected model: {best_model_name}")

# Export pipeline: preprocessing + model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])

joblib.dump(pipeline, 'models/fraud_model.pkl')
print("Pipeline saved to models/fraud_model.pkl")
