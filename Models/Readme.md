## 📦 Models Folder – README

This folder contains the notebooks and assets for training, evaluating, and analyzing machine learning models for predicting sales performance using the Superstore dataset.

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
| Model             | R²    | RMSE    | MAE     |
|------------------|-------|---------|---------|
| Linear Regression | 0.038 | 753.649 | 199.070 |
| Random Forest     | 0.591 | 491.690 | 85.937  |

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

📁 Top 20 worst RF predictions exported to:  
`Models/Artifacts/top_rf_prediction_errors.csv`

---

### 📁 Folder Structure

Models/
├── 03_model_training.ipynb
├── 04_model_evaluation.ipynb
├── model_metrics_log.csv
├── Artifacts/
│ ├── random_forest_model.pkl
│ ├── linear_regression_model.pkl
│ └── top_rf_prediction_errors.csv
### 📘 05_hyperparameter_tuning.ipynb

#### 🎯 Goal
Optimize the Random Forest model using `GridSearchCV` to improve predictive accuracy on sales.

#### ✅ Key Tasks
- Load the feature-engineered dataset
- Use `GridSearchCV` with 5-fold cross-validation
- Tune hyperparameters:
  - `n_estimators`: [100, 200]
  - `max_depth`: [None, 10, 20]
  - `min_samples_split`: [2, 5]
  - `min_samples_leaf`: [1, 2]
- Evaluate the best estimator on the test set
- Save the optimized model and full tuning results

#### 📊 Visualization
A heatmap of cross-validated R² scores was generated to show how model performance varies with tree depth and number of estimators.

#### 📤 Output
- ✅ Best model: `Models/Artifacts/random_forest_tuned.pkl`
- 📈 GridSearch results: `Models/Artifacts/rf_gridsearch_results.csv`
