# src/preprocessing.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def get_preprocessor():
    """
    Returns a ColumnTransformer for preprocessing the transaction data
    """
    # Numeric features
    numeric_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

    # Categorical features
    categorical_features = ['type']

    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    return preprocessor


def preprocess_data(input_path='data/raw/transactions.csv', output_path='data/processed/transactions_clean.csv'):
    """
    Load raw data, handle missing values, and save processed CSV
    """
    df = pd.read_csv(input_path)

    # Drop columns that cause data leakage
    if 'isFlaggedFraud' in df.columns:
        df = df.drop(columns=['isFlaggedFraud'])

    # Handle missing values
    df = df.fillna(0)

    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    preprocess_data()
