import shap
from joblib import load

# Load your test data here
import pandas as pd
X_test = pd.read_csv('data/X_test.csv')  # Update the path as needed

model = load('models/fraud_model.pkl')
explainer = shap.Explainer(model.named_steps['clf'], model.named_steps['pre'].transform(X_test))
shap_values = explainer(model.named_steps['pre'].transform(X_test))
shap.summary_plot(shap_values, features=model.named_steps['pre'].transform(X_test), max_display=3)
