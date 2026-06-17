import csv
import os
import random
from datetime import datetime, timedelta

def generate_mock_sales_csv(file_path: str, num_days: int = 730):
    """
    Generates a realistic mock sales dataset with weekly/monthly seasonality,
    price elasticity, promotion impacts, and holiday spikes.
    
    Parameters:
    - file_path: Path where the CSV will be saved.
    - num_days: Number of days of historical data to generate (default: 2 years).
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Categories and base prices
    categories = {
        "Electronics": {"base_price": 500.0, "base_sales": 15, "price_var": 100},
        "Apparel": {"base_price": 45.0, "base_sales": 40, "price_var": 15},
        "Home": {"base_price": 120.0, "base_sales": 25, "price_var": 30},
        "Beauty": {"base_price": 25.0, "base_sales": 50, "price_var": 8},
        "Sports": {"base_price": 85.0, "base_sales": 20, "price_var": 20}
    }
    
    stores = [1, 2, 3]
    
    # Standard holidays (simplified list of month-day tuples)
    holidays = {
        (1, 1),    # New Year's Day
        (7, 4),    # Independence Day
        (11, 25),  # Thanksgiving-ish (late Nov)
        (12, 24),  # Christmas Eve
        (12, 25),  # Christmas Day
        (12, 31)   # New Year's Eve
    }
    
    start_date = datetime.now() - timedelta(days=num_days)
    
    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Header
        writer.writerow(["date", "store_id", "product_category", "price", "promo", "holiday", "sales"])
        
        for day in range(num_days):
            current_date = start_date + timedelta(days=day)
            date_str = current_date.strftime("%Y-%m-%d")
            
            # Check if holiday
            is_holiday = 1 if (current_date.month, current_date.day) in holidays or current_date.weekday() == 0 and random.random() < 0.05 else 0
            
            for store in stores:
                # Randomize promo (approx 15% chance per day per store)
                is_promo = 1 if random.random() < 0.15 else 0
                
                for cat, config in categories.items():
                    # Base parameters
                    base_price = config["base_price"]
                    base_sales = config["base_sales"]
                    price_var = config["price_var"]
                    
                    # Calculate price: discounted if promo
                    price = base_price + random.uniform(-price_var, price_var)
                    if is_promo:
                        price = price * random.uniform(0.75, 0.90)  # 10% to 25% discount
                    price = round(max(5.0, price), 2)
                    
                    # Base sales for this category and store
                    sales = base_sales * random.uniform(0.8, 1.2)
                    
                    # Store multiplier
                    if store == 1:
                        sales *= 1.1
                    elif store == 2:
                        sales *= 0.9
                    # else store 3 is baseline
                    
                    # Price elasticity (higher price -> lower sales)
                    price_ratio = price / base_price
                    sales = sales * (1.5 - 0.7 * price_ratio)
                    
                    # Promotion boost
                    if is_promo:
                        sales *= random.uniform(1.3, 1.7)  # 30-70% boost
                        
                    # Holiday boost
                    if is_holiday:
                        sales *= random.uniform(1.4, 2.0)  # 40-100% boost
                        
                    # Day of week seasonality (higher sales on Friday, Saturday, Sunday)
                    weekday = current_date.weekday()
                    if weekday in [4, 5]: # Fri, Sat
                        sales *= random.uniform(1.2, 1.4)
                    elif weekday == 6: # Sun
                        sales *= random.uniform(1.0, 1.2)
                    else: # Mon-Thu
                        sales *= random.uniform(0.85, 0.95)
                        
                    # Monthly seasonality (Nov, Dec spikes)
                    month = current_date.month
                    if month in [11, 12]:
                        sales *= random.uniform(1.25, 1.5)
                    elif month in [1, 2]:
                        sales *= random.uniform(0.8, 0.95)
                        
                    # Add noise
                    sales = int(max(0, round(sales + random.normalvariate(0, 3))))
                    
                    writer.writerow([date_str, store, cat, price, is_promo, is_holiday, sales])

if __name__ == "__main__":
    # Test generation
    generate_mock_sales_csv("../data/sales.csv")
    print("Test generation successful.")
