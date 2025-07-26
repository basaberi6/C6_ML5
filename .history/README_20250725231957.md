# 📊 Superstore Sales Prediction Project

## 🚀 Project Overview

This project builds a machine learning regression model to predict sales performance for a global Superstore using historical order and product data. The goal is to uncover patterns and insights that help guide decision-making for sales strategy and operational planning.

---

## 🔁 Workflow Summary

1. **Data Cleaning**
   - Loaded dataset from: `Data/Raw/Sample_Superstore.csv`
   - Checked for and removed missing or inconsistent data

2. **Feature Engineering**
   - Converted date columns
   - Extracted date-based features (year, month, day)
   - Calculated shipping delay
   - One-hot encoded categorical variables

3. **Model Training**
   - Trained and compared:
     - Linear Regression
     - Random Forest Regressor
   - Evaluated with R², RMSE, MAE

4. **Model Evaluation**
   - Visualized residuals, actual vs predicted, and Q-Q plots
   - Exported top model errors for review

5. **Hyperparameter Tuning**
   - Applied `GridSearchCV` to fine-tune Random Forest
   - Best model saved to: `Models/Artifacts/random_forest_tuned.pkl`

---

## 📊 Results Summary

| Model             | R² Score | RMSE     | MAE     |
|------------------|----------|----------|---------|
| Linear Regression | 0.038    | 753.649  | 199.070 |
| Random Forest     | 0.591    | 491.690  | 85.937  |
| Tuned RF (CV)     | ~0.62    | ~Lower   | ~Lower  |

✅ Random Forest significantly outperforms Linear Regression.

---

## 📁 Repository Structure

```
C6_ML5/
├── Data/
│   ├── Raw/
│   │   └── Sample_Superstore.csv
│   └── Processed/
│       └── feature_engineered_superstore.csv
├── Models/
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   ├── 05_hyperparameter_tuning.ipynb
│   ├── model_metrics_log.csv
│   └── Artifacts/
│       ├── random_forest_model.pkl
│       ├── linear_regression_model.pkl
│       ├── random_forest_tuned.pkl
│       └── rf_gridsearch_results.csv
└── README.md
```

---

## 📌 Business Insights

- **Shipping delay** and **product category** are strong predictors of sales.
- Random Forest reveals non-linear relationships that Linear Regression misses.
- Feature importance highlights which variables most influence sales performance.

---

## 👩‍💻 Tech Stack

- Python, Pandas, scikit-learn, Seaborn, Matplotlib
- Jupyter Notebooks
- VS Code, Git, GitHub

---

## 📬 Contact

For questions or collaboration: [nadeli11]

