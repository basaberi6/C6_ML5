# 📘 teammate_model.ipynb – Model Training and Evaluation with Hyperparameter Tuning

> **Purpose:**  
This notebook is created by a teammate for the same project and is used to train a separate Random Forest model using a different set of hyperparameters. The results from this notebook can be compared with those from `03_model_training.ipynb` to assess the impact of tuning and feature choices on model performance. It supports team-based experimentation and performance benchmarking.

---

## ✅ Overview of Workflow

### 1. 📦 Import Libraries
Loads libraries like `pandas`, `numpy`, `matplotlib`, and key modules from `scikit-learn` for data manipulation, modeling, and visualization.

### 2. 📂 Load & Enhance Data
- Loads the feature-engineered dataset.
- Extracts time-based features (`Order_Month`, `Order_Weekday`) and encodes them cyclically using sine and cosine transformations to preserve temporal patterns.

### 3. 🧪 Train-Test Split & Pipeline Setup
- Performs train/test split on the dataset.
- Defines a pipeline that includes:
  - One-Hot Encoding of categorical features via `ColumnTransformer`.
  - A `RandomForestRegressor` model wrapped inside the pipeline for end-to-end processing.

### 4. 🔍 RandomizedSearchCV Hyperparameter Tuning
- Optimizes key model parameters (`n_estimators`, `max_depth`, `min_samples_leaf`, etc.) using randomized search.
- Applies cross-validation to select the best model based on training set performance.

### 5. 📈 Model Evaluation
- Evaluates the tuned model on both training and test datasets using:
  - R² Score  
  - Adjusted R²  
  - Root Mean Squared Error (RMSE)
- Prints final model performance and best hyperparameters for comparison.

### 6. 📊 Feature Importance Visualization
- Extracts feature importance scores from the final model.
- Visualizes the most important features to explain what drives predictions.
