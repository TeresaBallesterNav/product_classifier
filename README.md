# Product Classification

The technical assignment consists of building a model to classify products into their respective categories based on their features. Once the model is trained and evaluated, the next step is to build an API to allow easy and efficient testing of the model through HTTP requests.

## Table of Contents

1. [Product Classification](#product-classification)
2. [Project Structure](#project-structure)
3. [Usage](#usage)
4. [Dataset Description](#dataset-description)
5. [Model Overview](#model-overview)
6. [Questions & Answers](#questions--answers)
7. [Task Description](#task-description)
8. [Contact](#contact)

## Project Structure

product_classification_clean/
    data/ # Raw, processed and cleaned data (not committed to git)
	models/ # Trained models 
	notebooks/ # Jupyter notebooks for EDA and experimentation (not committed to git)
	src/ # Source code for data cleaning, processing and training
	utils/ # Helper function to read raw data
	Dockerfile # File for containerized solution
    main.py # API (Fastapi) entry point
	README.md # Project overview and documentation file (the one you are reading)
	requirements.txt # Python package dependencies

## Usage
### Docker
You can easily run the product classification API using Docker by following these steps:

1. Build the Docker image

```bash
docker build -t product_classifier_app .
```

This command builds the image using the Dockerfile and tags it as product_classifier_app.

2. Run the Docker container

```bash
docker run -p 8000:8000 product_classifier_app
```
This runs the container and maps the internal port 8000 of the FastAPI app to the local machine's port 8000.
You will run this command to run the app.

3. Access the API
Once the container is running, you can interact with the API from your browser or API client:

Base URL: http://localhost:8000

Interactive Docs (Swagger UI): http://localhost:8000/docs - Suggested for more interactive solution

    If you are in the Swager UI follow these steps to interact:
    1. Click 'Post' and then 'Try it out' 
    2. A white box will appeared where you can add the product data, for example:
        ```json
        {
        "title": "PURINA Dog Chow Adult Dog Food",
        "description": "Complete dry food for adult dogs with chicken flavor.",
        "brand": "Purina",
        "price": 24.99,
        "features": [
            "Balanced formula with essential vitamins",
            "Supports immune system",
            "Size: 14 kg"
        ]
        }
        ```
    3. Click 'Execute'
    4. The predicted class will appear in the 'Response body'. Following the previous example:
        ```json
        {
        "predicted_category": "Pet Supplies"
        }
        ```

Alternative Docs (ReDoc): http://localhost:8000/redoc


## Dataset Description

The dataset is a simplified version of [Amazon 2018](https://jmcauley.ucsd.edu/data/amazon/), only containing products and their descriptions.

The dataset consists of a jsonl file where each is a json string describing a product.

Example of a product in the dataset:
```json
{
 "also_buy": ["B071WSK6R8", "B006K8N5WQ", "B01ASDJLX0", "B00658TPYI"],
 "also_view": [],
 "asin": "B00N31IGPO",
 "brand": "Speed Dealer Customs",
 "category": ["Automotive", "Replacement Parts", "Shocks, Struts & Suspension", "Tie Rod Ends & Parts", "Tie Rod Ends"],
 "description": ["Universal heim joint tie rod weld in tube adapter bung. Made in the USA by Speed Dealer Customs. Tube adapter measurements are as in the title, please contact us about any questions you 
may have."],
 "feature": ["Completely CNC machined 1045 Steel", "Single RH Tube Adapter", "Thread: 3/4-16", "O.D.: 1-1/4", "Fits 1-1/4\" tube with .120\" wall thickness"],
 "image": [],
 "price": "",
 "title": "3/4-16 RH Weld In Threaded Heim Joint Tube Adapter Bung for 1-1/4&quot; Dia by .120 Wall Tube",
 "main_cat": "Automotive"
}
```

### Field description
- also_buy/also_view: IDs of related products
- asin: ID of the product
- brand: brand of the product
- category: list of categories the product belong to, usually in hierarchical order
- description: description of the product
- feature: bullet point format features of the product
- image: url of product images (migth be empty)
- price: price in US dollars (might be empty)
- title: name of the product
- main_cat: main category of the product

`main_cat` can have one of the following values:
```json
["All Electronics",
 "Amazon Fashion",
 "Amazon Home",
 "Arts, Crafts & Sewing",
 "Automotive",
 "Books",
 "Buy a Kindle",
 "Camera & Photo",
 "Cell Phones & Accessories",
 "Computers",
 "Digital Music",
 "Grocery",
 "Health & Personal Care",
 "Home Audio & Theater",
 "Industrial & Scientific",
 "Movies & TV",
 "Musical Instruments",
 "Office Products",
 "Pet Supplies",
 "Sports & Outdoors",
 "Tools & Home Improvement",
 "Toys & Games",
 "Video Games"]
```

[Download dataset](https://drive.google.com/file/d/1Zf0Kdby-FHLdNXatMP0AD2nY0h-cjas3/view?usp=sharing)

Data can be read directly from the gzip file as:
```python
def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)
```

## Model Overview
This project aims to classify Amazon products into relevant categories based on their textual attributes (title, description, brand, etc.). The process involved several well-structured stages to ensure data quality and robust model performance.

### Methodology Overview

Before modeling and deployment, the following steps were followed:

1. Data Cleaning

The raw dataset contained over 1 million entries, requiring extensive cleaning:

- Removed unnecessary columns and duplicates.
- Converted lists or arrays to strings for processing.
- Dropped rows with missing or empty critical text fields.
- Standardized and normalized textual fields like title, description, feature, and brand.
- Filtered product categories with very few entries and grouped the rare ones under "Other".
- Handled invalid or generic brand names by replacing them with NaN or standardized names like "Amazon".
- Applied advanced text cleaning: removing HTML, URLs, punctuation, and lowering casing.
- Addressed missing price values by imputing medians based on brand and global statistics.

2. Data Preprocessing and Training
The cleaned dataset was processed using a combination of TF-IDF vectorization for text features and scaling for numerical features. The features were selected after the EDA where we identified the key predictors for the product classification.

The data was divided into target variable and features:
    Target (y): The main_cat_grouped column.
    Features (X): ['title', 'description', 'feature', 'brand', 'price'].

We used a ColumnTransformer to apply different transformations:

- title, description, feature, brand → TfidfVectorizer with varying max features (5000, 5000, 3000, 1000 respectively).
- price → StandardScaler.

3. Model Selection
The model selection process involved several iterations and experiments, starting from baseline models to more complex alternatives. The goal was to find a classifier that could generalize well across a diverse set of product categories.

1. Initial Model Using main_cat as Target
The first version of the model used the main_cat field as the target variable. An SGDClassifier was trained using the raw category labels. However, the results showed significant instability and performance issues due to high class imbalance—many categories were underrepresented, which led to poor generalization on minority classes.

2. Improved Labeling Using main_cat_grouped
To address class imbalance, a grouped version of the categories—main_cat_grouped—was created, aggregating sparse classes into an "Other" category. This substantially improved performance:

- Category balance was much better, with "Other" acting as a catch-all for underrepresented groups.
- The model showed higher and more stable precision, recall, and F1 scores across multiple categories.
- Macro and weighted average scores improved, indicating more reliable classification across all classes, not just the dominant ones.

This led to the decision to continue with main_cat_grouped as the target variable.

3. Hyperparameter Tuning and Refinement
The next step involved hyperparameter tuning on the grouped-label model (SGDClassifier). While overall performance improved, some specific brand-related categories were still misclassified. Further experiments and refinements are planned to address this.

At this point, the best-performing and most stable version was the one trained using the default hyperparameters (with class_weight='balanced') on main_cat_grouped, as implemented in the preprocessing_training.py script. This model was selected as the final version for deployment.

4. Testing XGBoost with TF-IDF and Embeddings
As part of experimentation, an XGBoost model was also tested using main_cat_grouped:

- First with TF-IDF transformed inputs (similar to the SGD pipeline).
- Then by generating sentence embeddings for textual fields using SentenceTransformer.

While XGBoost performed reasonably well, it required more computational resources and added complexity to the preprocessing pipeline. For this technical assignment, the SGDClassifier remained the best trade-off between performance, interpretability, and efficiency.

### Final Model
For classification, we used SGDClassifier from scikit-learn, which is efficient for large-scale datasets.

Key configuration:
- loss='log_loss' (for probabilistic outputs).
- class_weight='balanced' (to handle class imbalance).
- max_iter=1000
- n_jobs=-1 (parallel processing)

#### Evaluation 

This section summarizes the performance of the final multi-class classification model.

**Key insights from the evaluation:**

- Overall accuracy: `78%` on the test set of 20,000 samples.
- Macro-average F1-score: `0.75`, which suggests reasonably balanced performance across all categories.
- Weighted-average F1-score: `0.78`, reflecting strong performance aligned with class distribution.


**Confusion Matrix Observations:**
- Most of the values are concentrated along the diagonal, indicating correct predictions.
- However:
  - ‘All Electronics’ has a low recall of `0.38`, meaning the model often misses this class.
  - ‘Other’ has both low precision (`0.33`) and low recall (`0.43`), showing that this catch-all class is often confused with others.

These results suggest that certain minority or generic classes are harder to predict reliably, likely due to data imbalance or overlapping features.

**Pros:**
- Good overall accuracy (78%) on a diverse and large dataset.
- High precision and recall for well-defined product categories like: Amazon Fashion (F1 = 0.96), Pet Supplies (F1 = 0.91) and Video Games (F1 = 0.89).
- Minimal class confusion, as shown by the sparse off-diagonal values in the confusion matrix.
- Scalable approach: The current pipeline can handle a large volume of samples efficiently.
- Modular design, allowing easy experimentation with different vectorization or model techniques (e.g. TF-IDF → sentence embeddings).

**Cons / Limitations:**
- Low recall for some categories, especially:
    - All Electronics (Recall = 0.38)
    - Other (Recall = 0.43)

- Poor precision for ambiguous classes, notably:
    - Other (Precision = 0.33), likely due to overlapping textual descriptions.
- TF-IDF limitations: While effective, this method doesn’t capture semantic meaning across different product titles and descriptions. A move to pretrained embeddings (e.g., SentenceTransformers) might improve this.

## Questions & Answers
Answer the following questions:
1. What would you change in your solution if you needed to predict all the categories?
    If I needed to predict the original main_cat categories (instead of the grouped main_cat_grouped), I would address the significant class imbalance and sparsity challenges.
    - Class balancing techniques: I would explore data augmentation or oversampling methods like SMOTE, or under-sampling the majority classes, to improve generalization for underrepresented categories
    - Hierarchical classification: Since many product categories are nested or hierarchical, I would explore a two-stage model: first predicting high-level categories (like in main_cat_grouped), and then a second model to predict the fine-grained main_cat.
    - Model architecture: I would experiment with more expressive models (e.g., deep learning classifiers like BERT-based models) to better handle nuanced class distinctions and text variability.

2. How would you deploy this API on the cloud?
    To deploy the API on the cloud, I would follow these steps:
    1. Choose a cloud provider: Use a platform like AWS, Google Cloud, or Azure.
    2. Containerize: Since the API is already Dockerized, I would push the image to a container registry (for example, Docker Hub).
    3. Deploy and scale: Use a managed container service to deploy the container.
    4. Monitoring: Use integrated tools to track performance, request errors, and latency.
    5. Security: Add authentication to protect the API (e.g., using API keys or OAuth2).
    6. CI/CD: Integrate GitHub Actions or similar tools for automatic testing and deployment.

3. If this model was deployed to categorize products without any supervision which metrics would you check to detect data drifting? When would you need to retrain?

    To monitor for data drift, I would:

    - Use statistical tests (e.g., KS test, Jensen-Shannon divergence) or tools like evidently to compare distributions of incoming data (e.g., TF-IDF embeddings or SentenceTransformer vectors) with training data.
    - Track confidence scores of model predictions. A significant drop in confidence over time may signal unfamiliar inputs.
    - Monitor if the frequency of predicted categories shifts significantly compared to training data.

    I would need to retrain in 2 situations:¡
    1. If manual reviews or shadow deployments indicate a drop in accuracy or precision/recall on key classes.
    2. Significant data drift: If monitored drift metrics exceed thresholds.


## Task description

- You should create a model that predicts `main_cat` using any of the other fields except `category` list. The model should be developed in python, you can use any pre-trained models and thirdparty 
libraries you need (for example huggingface).

- You should create a HTTP API endpoint that is capable of performing inference and return the predicted `main_cat` when receiving the rest of product fields.

- Both the training code (if needed) and the inference API should be dockerized and easy for us to run and test locally. **Only docker build and docker run commands should be necessary to perform training or setting up the inference API**.

- You should also provide a detailed analysis of the performance of your model. **In this test we're not looking for the best model performance but we expect a good understanding of your solution performance and it's limitations**.

- We will value:
    - Code cleanliness and good practices.
    - API design and architecture.
    - A clear understanding of the model performance, strengths and weak points.

 ## Contact
 For any further doubts or questions, please reach me out at: teresaballesternavarro@gmail.com