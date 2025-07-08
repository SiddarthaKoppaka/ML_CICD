import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer to add derived features like ratio
class FeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["income_to_loan_ratio"] = X["income"] / (X["loan_amount"] + 1e-6)
        return X


def save_features_to_store(input_path="data/raw/loan_data.csv", version="v1"):
    df = pd.read_csv(input_path)

    # Columns
    categorical_cols = ["employment_type"]
    numerical_cols = ["age", "income", "loan_amount", "loan_term_months", "credit_score"]

    # Preprocessing
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine all transformations
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    # Apply custom transformation first
    custom_features = FeatureAdder().fit_transform(df)

    # Fit & transform features
    transformed_features = preprocessor.fit_transform(custom_features)

    # Get transformed column names
    cat_feature_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
    final_feature_names = numerical_cols + list(cat_feature_names) + ["income_to_loan_ratio"]

    # Create final DataFrame
    import numpy as np
    processed_df = pd.DataFrame(transformed_features.toarray() if hasattr(transformed_features, 'toarray') else transformed_features,
                                columns=final_feature_names[:-1])  # all but income_to_loan_ratio

    # Add derived feature manually (since it wasn’t transformed)
    # Add derived feature manually (since it wasn’t transformed)
    processed_df["income_to_loan_ratio"] = custom_features["income_to_loan_ratio"].values

    # 🔧 ADD THIS TO INCLUDE TARGET
    processed_df["default"] = df["default"].values  

    # Save to feature store
    output_dir = Path("data/features/")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{version}_features.parquet"
    processed_df.to_parquet(output_path, index=False)

    print(f"[INFO] Processed features saved to {output_path}")


# Run
save_features_to_store()
