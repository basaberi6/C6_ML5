# 🧠 Final Project: Predicting the Determinants of Superstore Sales

![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Final--Submission-success)
![ML Model](https://img.shields.io/badge/model-RandomForest-orange)

> “Machines are learning — and so are we.” 💻📊

---

# Sales Profit Prediction

## Table of Contents
- [🛠️ Tech Stack](#️-tech-stack)
- [Project Objectives](#project-objectives)
- [Data Preparation & Feature Engineering](#data-preparation--feature-engineering)
- [Modeling Approach](#modeling-approach)
- [Best Model](#best-model)
- [📊 Best Model Results Snapshot](#📊-best-model-results-snapshot)
- [Business Value](#business-value)
- [Key Takeaways](#key-takeaways)

## 🛠️ Tech Stack
- 🐍 Python 3.12  
- 📊 Pandas, NumPy, Matplotlib  
- 🤖 Scikit‑learn, XGBoost, LightGBM, RandomForestRegressor  
- 📂 Jupyter Notebook  

## Project Objectives
- Predict each sale’s profit so the team can make smarter decisions on pricing, promotions, and stock levels.

## Data Preparation & Feature Engineering
- Smoothed out seasonal swings with sine/cosine date transforms  
- Tamed large values by applying log and square transforms to quantities  
- Created business‑centric metrics such as profit per unit, discount‑to‑quantity ratios, and customer/category averages to capture buying habits  

## Modeling Approach
- **Linear Regression & Ridge Regression**: Established baseline performance  
- **Random Forest & Gradient Boosting**: Captured non‑linear interactions  

## Best Model
- **Random Forest** delivered the highest hold‑out R², explaining roughly 78% of the variability in sales.  
- While the SMAPE score shows there’s still room to improve absolute error, the model provides clear directional insight into how sales respond to key predictors.  

## 📊 Best Model Results Snapshot
```Best Model
Model: Random Forest
Test RMSE: 269.58
Test R^2: 78.26%
Test Adjusted R^2: 77.96%
Test SMAPE: 31.66%

Team members:

Babak S.
Divita Narang
Foram Patel
Murad Ahmed
Nastaran Adeli
Tala Amiri
