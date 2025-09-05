
import sys
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Any, Dict
from src.predicting import predict_category

# Add the root of the project to PYTHONPATH at runtime
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

app = FastAPI()

class ProductInput(BaseModel):
    title: str
    description: str
    brand: str
    price: float
    features: list[str]

    class Config:
        extra = "ignore"
        
@app.post("/predict")
def predict_product_category(product: ProductInput):  
    """
    Input: dictionary that contains all the characteristics of an Amazon product.
    Output: dictionary with the predicted category of the inputed product.
    
    Endpoint receives raw product input,
    applies cleaning, processing, and prediction pipeline,
    and returns the predicted category.
    """
    predicted_category = predict_category(product.dict())

    return {"predicted_category": predicted_category}

if __name__ == "__main__":
    import uvicorn
    import webbrowser

    webbrowser.open("http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)
