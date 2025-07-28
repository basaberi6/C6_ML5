# 🧠 Final Project: Predicting the Determinants of Superstore Sales

![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Final--Submission-success)
![ML Model](https://img.shields.io/badge/model-RandomForest-orange)

> “Machines are learning — and so are we.” 💻📊

---
## Table of Contents
- [🛠️ Tech Stack](#️-tech-stack)
- [Project Objectives](#project-objectives)
- [Workflow Summary](#workflow_summary)
- [Feature Engineering Summary](#feature_engineering_summary)
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
This project aims to develop a machine learning model that predicts sales performance based on factors such as order date, shipping mode, customer segment, and product category. The model aims to generate actionable insights from a global superstore Sales dataset.

Understanding the key drivers of sales helps businesses make data-informed decisions that improve both efficiency and profitability. By identifying the influential factors on sales performance, the model enables:

a.	Improved Demand Forecasting
Anticipate future sales trends to optimize inventory levels and reduce stockouts or overstocking.

b.	Operational Optimization
Evaluate the impact of shipping modes and timelines on sales to improve logistics strategies.

c.	 Product and Category Insights
Understand which product categories drive sales, enabling better assortment planning and promotional focus.
## Data Preparation & Feature Engineering
- Smoothed out seasonal swings with sine/cosine date transforms  
- Tamed large values by applying log and square transforms to quantities  
- Created business‑centric metrics such as profit per unit, discount‑to‑quantity ratios, and customer/category averages to capture buying habits  

### Key Business Takeaway From This Modeling Project for Sales Strategy
Sales performance is most strongly influenced by profit margins, not just discounts. While offering discounts can help, our analysis shows that excessive discounts often reduce profitability without significantly boosting sales. High-margin products like Copiers consistently deliver better sales outcomes. To grow revenue effectively, focus on promoting profitable products, optimizing discount strategies, and understanding customer buying patterns.

## Superstore Sales Dataset
Superstore dataset containing Information related to Sales, Profits and other interesting facts of a Superstore giant from 2015 to 2019.
https://www.kaggle.com/datasets/vivek468/superstore-dataset-final?resource=download

| Feature Name    | Type    | Distinct/Missing Values |
|----------------|---------|--------------------------|
| Row_ID         | numeric | 9800 distinct values     |
| Order_ID       | string  | 4922 distinct values     |
| Order_Date     | string  | 1230 distinct values     |
| Ship_Date      | string  | 1326 distinct values     |
| Ship_Mode      | string  | 4 distinct values        |
| Customer_ID    | string  | 793 distinct values      |
| Customer_Name  | string  | 793 distinct values      |
| Segment        | string  | 3 distinct values        |
| Country        | string  | 1 distinct values        |
| City           | string  | 529 distinct values      |
| State          | string  | 49 distinct values       |
| Postal_Code    | numeric | 626 distinct values      |
| Region         | string  | 4 distinct values        |
| Product_ID     | string  | 1861 distinct values     |
| Category       | string  | 3 distinct values        |
| Sub-Category   | string  | 17 distinct values       |
| Product_Name   | string  | 1849 distinct values     |
| Sales          | numeric | 5757 distinct values     |

## Workflow Summary
1. **Data Cleaning and Exploratory Data Analysis (EDA)**
   - Load dataset from: `Data/Raw/Sample_Superstore.csv`
   - Check for and removed missing or inconsistent data
   - Explore key distributions (Sales, Profit, Discount, etc.)
   - Visualize how Discount relates to Sales & Profit
   - Compare average Sales and Profit across Sub-Categories
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
6. **Comparison of two models from teammates**
   - Comparing performance Metrics of two models: R² Score , RMSE,  Adjusted R²,  MAPE / SMAPE  
   - Summary table of both model's results: training and test scores are displayed side by side for easy comparison.
   - Visual Comparisons:
      - Bar chart of key metrics (R², RMSE, Adjusted R²) comparing models.
      - Feature Importance Bar Plots for both models, highlighting which factors drive predictions.
      - Scatter plots of actual vs. predicted sales for each model to visually assess how close predictions are to reality.
      - Top Feature Drivers: Extracts and displays the most important predictive features for each model to identify any shifts in model behavior.

## Feature Engineering Summary

After engineering, the dataset includes:

- Sales (target)
- Numerical Features Quantity, Discount, Profit (predictors/ important indicators of order size, pricing strategy, and margin)
- Engineered features :
      - Date-based features from 'Order Date' (Order Year, Order Month, Order Week, Order Weekday) to capture seasonal and weekly sales patterns
      - Shipping delay = Ship date - Order Date as proxy for customer experience, delivery efficiency, and potential impact on Sales or Profit
- One-hot encoded categories (Ship Mode, Segment, Region, Category, Sub-Category)
      - Converting categorical values to binary flags for model readability
      - Numeric representation of day-of-week trends

High-Cardinality Text/IDs (Customer Name, Product Name, Order ID, Product ID, etc.) are dropped from dataset to avoid overfitting and unnecessary complexity

Summary:

| Feature                          | Type            | Why It Was Included                         |
| -------------------------------- | --------------- | ------------------------------------------- |
| `Shipping_Delay`                 | Numeric         | Customer experience / operations proxy      |
| `Order_Weekday`                  | Ordinal         | Temporal buying patterns                    |
| `Order_Year/Month`               | Categorical-ish | Seasonality / yearly trends                 |
| `Discount`, `Profit`, `Quantity` | Numeric         | Price/margin context for sales              |
| Categorical dummies              | Binary          | Segment, region, product-level segmentation |      

## Modeling Approach
- **Linear Regression & Ridge Regression**: Established baseline performance  
- **Random Forest & Gradient Boosting**: Captured non‑linear interactions
- **Random Forest significantly outperforms Linear Regression.
  

## Best Model
Two models built using Random Forest Regressors and included structured feature engineering and hyperparameter tuning were compared. With the same modeling approach, the key distinction lies in the **depth of feature construction**.

## 📊 Best Model Results Snapshot

| Model                    | R² (Test) | RMSE (Test) | MAE (Test) |
|------------------------- |-----------|-------------|------------|
| **[nadeli11] Model**     | 0.591     | 491.690     | 85.937     |
| **[MuradAhmed00] Model** | 0.7828    | 269.45      | 75.16      |

### 🔬 Key Differences
- **[nadeli11] model** utilized foundational engineered features like `Profit`, `Discount`, `Order_Week`, and one-hot encoded categories.
- **[MuradAhmed00] model** extended feature engineering to include composite and aggregated variables such as `Profit_per_Unit`, `Discount_to_Qty`, `Cust_Order_Count`, and statistical measures at category/sub-category levels.

### 📈 Interpretation
- The [MuradAhmed00]’s model achieved **higher predictive accuracy**, reflected by a 32% gain in R² and a lower RMSE on test data.
- [nadeli11] model showed solid performance and interpretable results, particularly highlighting `Profit` as the dominant predictor of `Sales`.
- **Profit consistently drives sales** across both models, validating a strategy that focuses on maintaining high-margin product lines.
- The [MuradAhmed00]’s model reveals that **advanced customer and discount analytics** may offer added lift — useful for fine-tuning promotional and pricing strategies.
- [nadeli11]] remains more **interpretable and operationally lightweight**, making it suitable as a baseline or in resource-constrained deployments.

### 💡 Business Implications
Sales performance is most strongly influenced by profit margins, not just discounts. While offering discounts can help, our analysis shows that excessive discounts often reduce profitability without significantly boosting sales. High-margin products like Copiers consistently deliver better sales outcomes. To grow revenue effectively, focus on promoting profitable products, optimizing discount strategies, and understanding customer buying patterns.

## Team members:

Babak S. [basaberi6]
Divita Narang [divitaN-dev] Video: https://www.loom.com/share/1402e4bb1f084e5cbaaa179d16b6c161?sid=a7decfff-5e27-492a-8bca-99706fb2c72b
Foram Patel [Foram2248]
Murad Ahmed [MuradAhmed00] Video: https://drive.google.com/file/d/1Qzo7YNhP6uxSO0bUqlg0PiFXF7MwUxa6/view?usp=drive_link
Nastaran Adeli [nadeli11]
Tala Amiri [Talaamiri]
