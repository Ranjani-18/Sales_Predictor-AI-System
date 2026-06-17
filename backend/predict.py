import os
import pandas as pd
import joblib

try:
    from backend.feature import add_date_features, get_feature_columns
except ImportError:
    from feature import add_date_features, get_feature_columns

_model_cache = None
_last_model_path = None

def load_model_data(model_path: str):
    """
    Loads and caches the trained model data from disk.
    """
    global _model_cache, _last_model_path
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Train the model first.")
        
    if _model_cache is None or _last_model_path != model_path:
        _model_cache = joblib.load(model_path)
        _last_model_path = model_path
        
    return _model_cache

def predict_sales_inference(input_data: dict, model_path: str) -> float:
    """
    Predicts sales for a single dictionary of inputs.
    
    Expected keys:
    - date (str YYYY-MM-DD)
    - store_id (int)
    - product_category (str)
    - price (float)
    - promo (int, 0 or 1)
    - holiday (int, 0 or 1)
    """
    model_data = load_model_data(model_path)
    pipeline = model_data["pipeline"]
    
    # Convert input to DataFrame
    df = pd.DataFrame([input_data])
    
    # Engineer date features
    df = add_date_features(df)
    
    # Select feature columns in correct order
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    
    # Predict
    prediction = pipeline.predict(X)
    
    return float(prediction[0])
