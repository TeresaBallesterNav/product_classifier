import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import html
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

## Load Data ##
def load_data(filepath, sample_size=None):
    df = pd.read_parquet(filepath)
    if sample_size:
        df = df.iloc[:sample_size]
    return df

## Utility functions ##
def is_empty(val):
    try:
        if val in ("", [], {}, None):
            return True
        if hasattr(val, "size") and val.size == 0:
            return True
    except:
        pass
    return False

def convert_list_or_array_to_string(val):
    if isinstance(val, (list, np.ndarray)):
        return ' '.join(str(x) for x in val)
    elif pd.isna(val):
        return ''
    return str(val)

def clean_text(text):
    if not isinstance(text, str):
        return "unknown"
    text = text.lower() # All to lower case
    text = re.sub(r'[^a-z0-9\s]', ' ', text) # Remove non-alphabetic charcaters
    text = re.sub(r'\s+', ' ', text) # Remove multiple spaces
    return text.strip()

def clean_description(text):
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = ' '.join([word for word in text.split() if len(word) > 2])
    return text.strip()

def deep_clean(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text) # Decode HTML entities
    text = BeautifulSoup(text, "html.parser").get_text(" ") # Remove HTML tags
    text = re.sub(r"http\S+|www\S+", "", text)# Remove URLs
    words = text.split()  # Split into words
    words = [word for word in words if word.lower() not in ENGLISH_STOP_WORDS]  # Remove stopwords
    return ' '.join(words).strip()

def extract_price(text):
    if isinstance(text, str):
        match = re.search(r"\d+(\.\d{1,2})?", text)
        if match:
            return float(match.group())
    return np.nan


## Cleaning Data ##

def clean_dataframe(df):

    # Drop unnecessary columns
    df = df.drop(columns=['category', 'image', 'also_buy', 'also_view'], errors='ignore')

    # Drop duplicates based on 'asin'
    df = df.drop_duplicates(subset='asin').copy()

    #Convert numpy arrays to str
    for col in ['description', 'feature']:
        df[col] = df[col].apply(convert_list_or_array_to_string)
    
    # Drop rows with empty or null description/title/feature
    df = df[~df[['description', 'title', 'feature']].map(is_empty).any(axis=1)]

    # Set index for traceability
    df = df.set_index("asin")

    # Filter by 'main_cat'
    cat_counts = df["main_cat"].value_counts()
    valid_cats = cat_counts[cat_counts >= 100].index
    df = df[df["main_cat"].isin(valid_cats)].copy()
    cat_group = df["main_cat"].value_counts()
    to_other = cat_group[(cat_group < 2000)].index
    df["main_cat_grouped"] = df["main_cat"].apply(lambda x: "Other" if x in to_other else x)

    # Normalize text (function defined previously)
    for col in ['title',  'brand']:
        df[col] = df[col].astype(str).apply(clean_text)

    # Deep clean description (function defined previously)
    df["description"] = (
        df["description"]
                .apply(deep_clean)     # Remove HTML, URLs, etc.
                .apply(clean_text)     # Lowercase + strip punctuation
        )
    # Drop duplicate descriptions
    df = df.drop_duplicates(subset=["description"]).copy()

    # Drop duplicates of title + brand
    df = df[~df[['brand']].map(is_empty).any(axis=1)].copy()
    dup_mask = df.duplicated(subset=["title", "brand"], keep=False)
    df = df[~dup_mask].copy()

    # Filter short titles, and drop those that do not give enough information
    df["title_len"] = df["title"].apply(lambda x: len(x.split()))
    df = df[df["title_len"] > 1].drop(columns=["title_len"])

    # Standardize brand again (function defined previously)
    df["brand"] = df["brand"].apply(clean_text)

    # Replace unknown/generic/empty brand strings with NaN
    invalid_brands = {"", "unknown", "generic", "na", "n a", "none", "no brand", "nan"}
    df["brand"] = df["brand"].apply(lambda x: np.nan if x.strip() in invalid_brands else x)

    # Rename Amazon brands
    brands_with_amazon = df["brand"].dropna().unique()
    brands_with_amazon = [b for b in brands_with_amazon if "amazon" in b.lower()]
    df.loc[df["brand"].isin(brands_with_amazon), "brand"] = "Amazon"
    
    # Drop rows where 'feature' length is less than or equal to 1
    df["feature_len"] = df["feature"].apply(lambda x: len(x.split()))
    df = df[df["feature_len"] > 1].drop(columns=["feature_len"])
    
    # Deep clean feature
    df["feature"] = (
        df["feature"]
                .apply(deep_clean)     # Remove HTML, URLs, etc.
                .apply(clean_text)     # Lowercase + strip punctuation
        )

    # Convert 'price' column to numeric
    df['price'] = df['price'].apply(extract_price)

    # Handle missing price values
    # Remeber which prices were misisng
    df["price_missing"] = df["price"].isna().astype(int)
    # Grouping the dataframe by 'brand', and let´s fill per brand group the NaN values with the median
    df["price"] = df.groupby("brand")["price"].transform(
        lambda x: x.fillna(x.median())
    )
    # With the rest of NaN let´s fill it with the global median
    df['price'] = df['price'].fillna(df['price'].median())
    
    # Ensure text columns are filled and of type str
    text_columns = ['title', 'description', 'feature', 'brand']
    for col in text_columns:
        df[col] = df[col].fillna("").astype(str)

    return df


## Run ##
if __name__ == "__main__":
    from pathlib import Path
    import os

    # Project routes
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed_data"
    output_dir = project_root / "data" / "cleaned_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all chunks of data
    chunk_files = sorted(processed_dir.glob("products_chunk_*.parquet"))

    for i, chunk_path in enumerate(chunk_files):
        print(f"Cleaning chunk {chunk_path.name}")
        df = load_data(chunk_path)
        df_clean = clean_dataframe(df)
        cleaned_path = output_dir / f"cleaned_chunk_{i}.parquet"
        df_clean.to_parquet(cleaned_path, index=False)
        print(f"Saved: {cleaned_path}")
