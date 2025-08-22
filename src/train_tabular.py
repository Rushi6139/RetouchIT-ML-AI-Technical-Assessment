import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Load preprocessed dataset
data_path = "data/processed/transactions_clean.csv"
df = pd.read_csv(data_path)

print("✅ Data loaded for training")
print("Shape:", df.shape)

# -----------------------------
# Step 1: Define features + target
# -----------------------------
# Change "isFraud" to your dataset's fraud label column
target_col = "isFraud"  

if target_col not in df.columns:
    raise ValueError(f"Column '{target_col}' not found in dataset. Available: {df.columns.tolist()}")

X = df.drop(columns=[target_col])
y = df[target_col]

# -----------------------------
# Step 2: Encode categorical variables
# -----------------------------
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# -----------------------------
# Step 3: Scale numerical features
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# Step 4: Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Step 5: Train Logistic Regression Model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# Step 6: Evaluate
# -----------------------------
y_pred = model.predict(X_test)

print("\n✅ Model Training Complete")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Step 7: Save Model
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fraud_tabular_model.pkl")
print("\n💾 Model saved at: models/fraud_tabular_model.pkl")
