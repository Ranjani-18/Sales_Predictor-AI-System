# Sales Predictor AI System

An AI-powered sales prediction system designed to build, train, and forecast future demand. Built with a FastAPI backend running a Random Forest Regression model and a premium dark-mode glassmorphism dashboard.

## Features
- **Data Preprocessing**: Standardized cleanup of transactional records, handling missing values, date alignment, and chronological sorting.
- **Feature Engineering**: Decomposes date metrics to extract annual seasonality (`month`) and weekly peaks (`day_of_week`, `is_weekend`).
- **Machine Learning**: Fits a Random Forest Regression model with chronological train/test split. Logs performance indicators ($R^2$, MAPE, MAE, RMSE) and exports feature importances.
- **REST API**: Exposes JSON endpoints for uploading data, model training, prediction, and retrieving model status.
- **Visual UI**: Interactive dashboard utilizing Chart.js, file drag-and-drop, range sliders, and active switches to simulate live scenario planning.

## Tech Stack
- **Backend**: Python, FastAPI, Uvicorn, Pandas, Scikit-Learn, Joblib
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (ES6), Chart.js, FontAwesome

## Getting Started

1. **Install Python dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Run the FastAPI server**:
   ```bash
   python -m uvicorn backend.app:app --reload
   ```

3. **Open the interface**:
   Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.
