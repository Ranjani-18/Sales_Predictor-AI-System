import os
import pandas as pd

def clean_data(input_path: str, output_path: str) -> dict:
    """
    Cleans raw sales data: parses dates, handles missing values, sorts chronologically,
    and saves the cleaned data.
    
    Returns a dictionary of summary statistics.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")
        
    # Read the dataset
    df = pd.read_csv(input_path)
    
    # Check required columns
    required_cols = {"date", "store_id", "product_category", "price", "promo", "holiday", "sales"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")
        
    # Convert date and sort
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date").reset_index(drop=True)
    
    # Handle missing values
    # For price: fill with median price for that category
    if df["price"].isnull().any():
        df["price"] = df.groupby("product_category")["price"].transform(lambda x: x.fillna(x.median()))
        
    # Fill sales with 0 if missing
    df["sales"] = df["sales"].fillna(0).astype(int)
    
    # Fill binary flags
    df["promo"] = df["promo"].fillna(0).astype(int)
    df["holiday"] = df["holiday"].fillna(0).astype(int)
    df["store_id"] = df["store_id"].astype(int)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save processed dataset
    df.to_csv(output_path, index=False)
    
    # Prepare summary statistics
    summary = {
        "total_records": len(df),
        "start_date": df["date"].min().strftime("%Y-%m-%d"),
        "end_date": df["date"].max().strftime("%Y-%m-%d"),
        "unique_stores": df["store_id"].nunique(),
        "unique_categories": df["product_category"].nunique(),
        "categories_list": list(df["product_category"].unique()),
        "average_sales": float(df["sales"].mean()),
        "average_price": float(df["price"].mean())
    }
    
    return summary

if __name__ == "__main__":
    # Test clean_data
    try:
        stats = clean_data("../data/sales.csv", "../data/processed_sales.csv")
        print("Preprocessing test successful. Summary stats:")
        print(stats)
    except Exception as e:
        print(f"Error: {e}")
