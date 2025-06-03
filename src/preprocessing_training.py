import pandas as pd
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

## Load cleaned data ##
def load_cleaned_data(cleaned_folder):
    path = Path(cleaned_folder)
    chunk_paths = sorted(path.glob("cleaned_chunk_*.parquet"))
    df_chunks = [pd.read_parquet(chunk) for chunk in chunk_paths]
    return pd.concat(df_chunks, ignore_index=True)

## Preprocess cleaned data to train later ##
if __name__ == "__main__":
    # Define project routes and directories
    project_root = Path(__file__).resolve().parents[1]
    cleaned_data_dir = project_root / "data" / "cleaned_data"          
    models_dir = project_root / "model"                                    
    models_dir.mkdir(exist_ok=True)   

    # Load data already cleaned 
    df_clean = load_cleaned_data(cleaned_data_dir)

    # Define target variable (y) and predictors (X)
    X = df_clean[['title', 'description', 'feature', 'brand', 'price']]
    y = df_clean['main_cat_grouped']

    # Transform text variables and apply standard scale to the numerical variable (price)
    preprocessor = ColumnTransformer(
        transformers=[
            ('title_tfidf', TfidfVectorizer(max_features=5000), 'title'),
            ('desc_tfidf', TfidfVectorizer(max_features=5000), 'description'),
            ('feature_tfidf', TfidfVectorizer(max_features=3000), 'feature'),
            ('brand_tfidf', TfidfVectorizer(max_features=1000), 'brand'),
            ('scaler', StandardScaler(), ['price'])
        ],
        remainder='drop'
    )

    # Define model
    model = SGDClassifier(loss='log_loss', class_weight='balanced', max_iter=1000, n_jobs=-1)

    # Combine preprocessor + model in a pipeline
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', model)
    ])

    # Fit model pipeline on X and y
    pipeline.fit(X, y)

    # Save the fully trained pipeline for later prediction
    joblib.dump(pipeline, models_dir / "model_pipeline.joblib")
    
    print("Model trained and saved in 'models/model_pipeline.joblib'.")
