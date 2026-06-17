import pandas as pd
import numpy as np

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts chronological and seasonal features from the 'date' column.
    Expected columns: df['date'] must be datetime-like.
    """
    df = df.copy()
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
        
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek # 0 = Monday, 6 = Sunday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    return df

def get_feature_columns():
    """
    Returns the list of features expected by the machine learning model.
    """
    return ["store_id", "product_category", "price", "promo", "holiday", "year", "month", "day", "day_of_week", "is_weekend"]
