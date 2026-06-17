import os
import shutil
import pandas as pd
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.preprocess import clean_data
from backend.train import train_sales_model
from backend.predict import predict_sales_inference
from utils.helper import generate_mock_sales_csv

app = FastAPI(
    title="Sales Predictor API",
    description="AI-powered Sales Prediction System",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FRONTEND_DIR, exist_ok=True)

# Define Pydantic request models
class PredictionInput(BaseModel):
    date: str = Field(..., description="Date of prediction in YYYY-MM-DD format")
    store_id: int = Field(..., description="Store Identifier (e.g. 1, 2, 3)")
    product_category: str = Field(..., description="Product Category (Electronics, Apparel, Home, Beauty, Sports)")
    price: float = Field(..., description="Unit price of the item")
    promo: int = Field(..., description="Is promo active (0 or 1)")
    holiday: int = Field(..., description="Is holiday (0 or 1)")

@app.post("/generate-mock-data")
def generate_mock_data(days: int = 730):
    """
    Generates a realistic mock dataset, preprocesses it, and prepares it for training.
    """
    try:
        raw_path = os.path.join(DATA_DIR, "sales.csv")
        processed_path = os.path.join(DATA_DIR, "processed_sales.csv")
        
        # Generate raw mock data
        generate_mock_sales_csv(raw_path, num_days=days)
        
        # Automatically preprocess raw data
        summary = clean_data(raw_path, processed_path)
        
        return {
            "success": True,
            "message": f"Generated {days} days of sales data successfully.",
            "data_summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    """
    Uploads a custom sales.csv file and automatically runs data preprocessing.
    """
    try:
        raw_path = os.path.join(DATA_DIR, "sales.csv")
        processed_path = os.path.join(DATA_DIR, "processed_sales.csv")

        # Save uploaded file
        with open(raw_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Preprocess uploaded data
        summary = clean_data(raw_path, processed_path)

        return {
            "success": True,
            "message": "Dataset uploaded and preprocessed successfully.",
            "file": file.filename,
            "data_summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train-model")
def train_model():
    """
    Triggers model training on the processed sales dataset.
    """
    processed_path = os.path.join(DATA_DIR, "processed_sales.csv")
    model_path = os.path.join(MODEL_DIR, "sales_model.pkl")

    if not os.path.exists(processed_path):
        raise HTTPException(
            status_code=400,
            detail="No processed dataset found. Upload a dataset or generate mock data first."
        )

    try:
        metrics = train_sales_model(processed_path, model_path)
        
        # Read back trained metadata
        model_data = joblib.load(model_path)
        
        return {
            "success": True,
            "message": "Model trained successfully.",
            "metrics": model_data["metrics"],
            "feature_importances": model_data["feature_importances"][:5], # top 5
            "trained_date": model_data["trained_date"],
            "training_size": model_data["training_size"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-sales")
def predict_sales(input_data: PredictionInput):
    """
    Accepts a single day's feature dict and predicts expected unit sales.
    """
    model_path = os.path.join(MODEL_DIR, "sales_model.pkl")

    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail="Model not found. Train the model first."
        )

    try:
        # Convert Pydantic object to dictionary
        input_dict = input_data.model_dump()
        
        predicted_sales = predict_sales_inference(input_dict, model_path)

        return {
            "success": True,
            "predicted_sales": max(0.0, round(predicted_sales, 2))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-status")
def get_model_status():
    """
    Retrieves performance and training details of the current model.
    """
    model_path = os.path.join(MODEL_DIR, "sales_model.pkl")
    processed_path = os.path.join(DATA_DIR, "processed_sales.csv")
    
    status = {
        "dataset_exists": os.path.exists(processed_path),
        "model_exists": os.path.exists(model_path),
        "model_metadata": None
    }
    
    if status["dataset_exists"]:
        # Get count of processed records
        try:
            df = pd.read_csv(processed_path)
            status["dataset_records"] = len(df)
        except Exception:
            status["dataset_records"] = 0
            
    if status["model_exists"]:
        try:
            model_data = joblib.load(model_path)
            status["model_metadata"] = {
                "metrics": model_data["metrics"],
                "feature_importances": model_data["feature_importances"][:5],
                "trained_date": model_data["trained_date"],
                "training_size": model_data["training_size"]
            }
        except Exception as e:
            status["model_error"] = str(e)
            
    return status

# Serve frontend static assets
# If folders/files in FRONTEND_DIR exist, serve them.
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def read_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)


def _open_browser(url: str, delay_seconds: float = 1.5):
    import threading
    import webbrowser

    def _try_open():
        try:
            webbrowser.open_new_tab(url)
        except Exception:
            pass

    timer = threading.Timer(delay_seconds, _try_open)
    timer.daemon = True
    timer.start()


if __name__ == "__main__":
    import uvicorn

    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"

    print(f"Starting Sales Predictor AI server at {url}")
    _open_browser(url)

    uvicorn.run(app, host=host, port=port, reload=True)