import joblib
import pandas as pd
from pathlib import Path

# Define project root to locate the model file easily
project_root = Path(__file__).resolve().parents[1]

# Load your trained pipeline (preprocessing + model)
model_path = project_root / "models" / "model_pipeline.joblib"
pipeline = joblib.load(model_path)

MODEL_FEATURES = ["title", "description", "feature", "brand", "price"]

def fill_empty_fields(data: dict) -> dict:
    for key in ['title', 'description', 'feature', 'brand']:
        if key not in data or data[key] is None:
            data[key] = ""
    return data


def predict_category(product_data: dict) -> str:
    """
    Given a single product dictionary, make a prediction of the category.
    
    Args:
        product_data (dict): Raw product info with keys like 'title', 'description', etc.
    
    Returns:
        str: Predicted category label.
    """
    # Fill empty or missing text fields
    product_data = fill_empty_fields(product_data)
    
    # Keep only the fields that the model uses
    filtered_data = {key: product_data.get(key, None) for key in MODEL_FEATURES}
    
    # Convert dict to DataFrame (pipeline expects DataFrame input)
    df = pd.DataFrame([filtered_data])

    # Use the loaded pipeline to predict
    prediction = pipeline.predict(df)[0]
    
    return prediction

