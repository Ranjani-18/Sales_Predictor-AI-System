import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Import feature engineering functions
try:
    from backend.feature import add_date_features, get_feature_columns
except ImportError:
    from feature import add_date_features, get_feature_columns

def train_sales_model(data_path: str, model_path: str) -> dict:
    """
    Trains a Random Forest Regressor on the processed sales data.
    Saves the entire pipeline (preprocessing + model) and returns evaluation metrics.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at: {data_path}")
        
    # 1. Load Clean Data
    df = pd.read_csv(data_path)
    
    # 2. Engineer Features
    df = add_date_features(df)
    
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df["sales"]
    
    # 3. Chronological Train/Test Split (80% Train, 20% Test)
    # Since date is sorted, we can split by index
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 4. Define Preprocessor (One-Hot Encoding for categories)
    categorical_features = ["product_category"]
    numeric_features = [col for col in feature_cols if col not in categorical_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ],
        remainder="passthrough" # leaves numeric features as they are
    )
    
    # 5. Build Training Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ]
    )
    
    # 6. Fit Model
    pipeline.fit(X_train, y_train)
    
    # 7. Evaluate Model
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate baseline mean absolute percentage error (MAPE) - handle division by zero
    y_test_np = y_test.to_numpy()
    mape = np.mean(np.abs((y_test_np - y_pred) / np.where(y_test_np == 0, 1, y_test_np))) * 100
    
    # Calculate feature importances
    # Get feature names after one-hot encoding
    cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    all_feature_names = encoded_cat_names + numeric_features
    
    importances = pipeline.named_steps["regressor"].feature_importances_
    feature_importances = sorted(
        [{"feature": name, "importance": float(imp)} for name, imp in zip(all_feature_names, importances)],
        key=lambda x: x["importance"],
        reverse=True
    )
    
    # 8. Save Pipeline & Metadata
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model_data = {
        "pipeline": pipeline,
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "mape": float(mape)
        },
        "feature_importances": feature_importances,
        "trained_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_size": len(X_train),
        "testing_size": len(X_test)
    }
    
    joblib.dump(model_data, model_path)
    
    return model_data["metrics"]

if __name__ == "__main__":
    try:
        metrics = train_sales_model("../data/processed_sales.csv", "../models/sales_model.pkl")
        print("Training successful! Metrics:")
        print(metrics)
    except Exception as e:
        print(f"Error during training: {e}")
