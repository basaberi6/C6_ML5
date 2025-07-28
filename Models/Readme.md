## 📦 Models Folder – README

This folder contains the notebooks and assets for training, evaluating, and analyzing machine learning models for predicting sales performance using the Superstore dataset.
03_model_training.ipynb is based on [nadeli11] model as the baseline and teammate_model.ipynb is based on [MuradAhmed00] that had the best performance.

---

### 📘 03_model_training.ipynb

#### 🎯 Goal
Train regression models to predict `Sales` using engineered features from the cleaned Superstore dataset.

#### ✅ Key Tasks
- Load processed dataset with engineered features
- Split data into training and testing sets
- Train two models:
  - Linear Regression
  - Random Forest Regressor
- Evaluate performance using:
  - R² Score
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
- Save trained models as `.pkl` files
- Log model metrics to `model_metrics_log.csv`

#### 📤 Output

03_model_training.ipynb baseline model output vs. teammate_model.ipynb as the Best model

| Model                             | R² (Test) | RMSE (Test) | MAE (Test) |
|-----------------------------------|-----------|-------------|------------|
| **[nadeli11] Baseline Model**     | 0.591     | 491.690     | 85.937     |
| **[MuradAhmed00] Best Model**     | 0.7828    | 269.45      | 75.16      |

📁 Models saved to: `Models/Artifacts/`  
📄 Metrics logged to: `Models/model_metrics_log.csv`

---

### 📘 04_model_evaluation.ipynb

#### 🎯 Goal
Compare the trained models using diagnostic and error analysis techniques.

#### ✅ Key Tasks
- Reload saved models
- Predict on the test set
- Generate visual diagnostics:
  - Residual histograms
  - Actual vs. Predicted scatter plots
  - Q-Q plots of residuals
  - Residuals vs. Predicted values
- Export worst-performing predictions (outliers)

#### 📊 Key Insights
- **Random Forest** produces tighter, more normally distributed residuals.
- **Linear Regression** shows greater error spread and more heteroscedasticity.
- Diagnostic plots confirmed the superiority of Random Forest for this dataset.

📁 Top 10 worst RF predictions exported to:  
`Models/Artifacts/top_rf_prediction_errors.csv`

---

### 📘 05_hyperparameter_tuning.ipynb

#### 🎯 Goal
Optimize the Random Forest model using `GridSearchCV` to improve predictive accuracy on sales.

#### ✅ Key Tasks
- Load the feature-engineered dataset
- Use `GridSearchCV` with 5-fold cross-validation
- Baseline Model Tune hyperparameters:
  - `n_estimators`: [100, 200]
  - `max_depth`: [None, 10, 20]
  - `min_samples_split`: [2, 5]
  - `min_samples_leaf`: [1, 2]
- Best Model Tune hyperparameters:
  - 'model__max_depth': 17
  - 'model__max_features': 0.7,
  - 'model__min_samples_leaf': 5,
  - 'model__n_estimators': 444

- Evaluate the best estimator on the test set
- Save the optimized model and full tuning results
