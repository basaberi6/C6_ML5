# 📊 Superstore Sales Prediction Project
## Business Value of the Sales Prediction Model
This project aims to develop a machine learning regression model that predicts sales performance based on factors such as order date, shipping mode, customer segment, and product category. The model is built using Python and is designed to generate actionable insights from the Superstore Sales dataset.
Why This Matters
Understanding the key drivers of sales helps businesses make data-informed decisions that improve both efficiency and profitability. By identifying how operational and customer-related factors influence revenue, the model enables:
   Improved Demand Forecasting
      Anticipate future sales trends to optimize inventory levels and reduce stockouts or overstocking.
   Customer Segmentation and Personalization
      Identify high-performing customer segments to focus marketing and retention efforts more effectively.
   Operational Optimization
      Evaluate the impact of shipping modes and timelines on sales to improve logistics strategies.
   Product and Category Insights
      Understand which product categories drive sales, enabling better assortment planning and promotional focus.
   Seasonal and Temporal Planning
      Uncover time-based sales trends to align promotions, staffing, and supply with peak periods.
Industries That Can Benefit
•	Retail & E-Commerce: Optimize product offerings, logistics, and marketing.
•	Consumer Goods & Wholesale: Improve forecasting and inventory planning.
•	Logistics & Fulfillment Providers: Align shipping methods with customer expectations and sales performance.
•	Marketing & BI Consulting Firms: Deliver more targeted, data-driven client strategies.

## 🚀 Project Overview

This project applied machine learning to identify the key drivers of sales performance for a global retail superstore. Based on sug-catgories of soled product, this superstore is in the market of office supplies with the main product being copiers. 
The model was trained on a four-year dataset containing over 9,000 transactions, with fields capturing order timing, shipping mode, customer segment, geography, product categories, discount levels, sales revenue, and profit.
A Random Forest regression model was trained and fine-tuned to predict sales. The best model achieved an R² score of 0.79 on test/validation data, demonstrating its ability to capture complex relationships within the data.

===================
## 🧭 Key Business Takeaway From This Modeling Project for Sales Strategy
Sales performance is most strongly influenced by profit margins, not just discounts. While offering discounts can help, our analysis shows that excessive discounts often reduce profitability without significantly boosting sales. High-margin products like Copiers consistently deliver better sales outcomes. To grow revenue effectively, focus on promoting profitable products, optimizing discount strategies, and understanding customer buying patterns.

The most important finding based on the best model results, is that that profit alone explained over 67% of the model’s predictive power, making it by far the dominant driver of sales. While this might sound intuitive, the quantitative confirmation supports a strategy focused on high-margin products. Copiers, for instance, emerged as the strongest performer in both revenue and profit, while categories like Tables, Bookcases, and Supplies had consistently negative margins despite reasonable sales volumes.
Discount, by contrast, contributed only ~1.5% to the model's predictions, and the data showed that sales were actually highest when no discount was applied. This challenges the widely held belief that deeper discounts are a reliable lever for increasing sales and suggests that discounting strategies may be eroding margin without delivering returns.
Results of another another models has also showed that shipping mode has minimal impact on sale. Although Standard Class had the longest order-to- ship time, sales levels remained comparable across fulfillment options — suggesting customers are tolerant of slower shipping when expectations are clear.

These insights were not merely the result of data visualization but were derived from a
trained machine learning model that assigned quantitative weight to each predictor. This approach provided data-driven clarity on what really drives sales and equips the
business with actionable priorities for pricing, product management, and promotional focus.

=====================
Model Limitations and ext Steps for analysis

It is important to know the business and accounting context behind data collections or any calculations applied in the raw data. 
Having business and accounting context behind how data are collected, or features are calculated is important for capturing nuances of how features can influence each other or trusting the generalized predictions of model for crafting a sound business strategy. 
For example, 'Profit data as provided in the raw dataset is actually a calculated feature of its own.  We do not know if calculation of the profit is simply based on sold price minus cost of goods sold or it has included operational or inventory costs. Adding operational costs or inventory costs of goods may change the importance of Discount feature.

Additionally integrating external data sources such as marketing activity, competitor pricing, or economic indicators model can help with the model reflating real-world influences on sales. 
Providing the ability to integrate this model with updated financial and accounting data and retraining model based on updated data from time to time can also help model to capture the possibility of the change in overall business environment, customers demands or expectation (for example impact of COVID pandemic in 2020).

To further improve the accuracy and business value of the sales prediction model, future analysis could explore how performance varies across different customer segments or product categories, allowing for more targeted strategies. Incorporating time-based patterns such as seasonality or quarterly trendscan help the model adjust to predictable sales cycles. Advanced techniques like Gradient Boosting or XGBoost may also enhance predictive performance by capturing more complex relationships in the data. Additionally, simplifying the model by removing less important features based after the results of initial modelling cmay also reduce overfitting and improve the power generalizition.

======================

## 📊 Superstore Sales Dataset

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

---

## 📁 Repository Structure

```
C6_ML5/
├── Data/
│   ├── Raw/
│   │   └── Sample_Superstore.csv
│   └── Processed/
│       ├── feature_engineered_superstore.csv
|       ├── cleaned_superstore.ipynb
|.      └──data_review.ipynb  
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
├── Experiments/
|   ├── 06_model_comparison.ipynb
|   └── teammate_model.ipynb
└── README.md
```
---
## 👩‍💻 Tech Stack

- Python, Pandas, scikit-learn, Seaborn, Matplotlib, joblib
- Jupyter Notebooks
- VS Code, Git, GitHub

---

## 🔁 Workflow Summary

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

--------------------------------------------
## 🧠 Feature Engineering Summary

For the Baseline Model [nadeli11], after engineering, the dataset includes:

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


For the Best Model, [MuradAhmed00], feature engineer was as follow
   - Smoothed out seasonal swings with sine/cosine date transforms
   - Tamed large values by applying log and square transforms to quantities
   - Created business‑centric metrics such as profit per unit, discount‑to‑quantity ratios, and customer/category averages to capture buying habits


## 📊 Modeling Approach Results

✅ For both Baseline Model and Best Model, Random Forest significantly outperforms Linear Regression.


---

## 🔍 Comparison of two different models & insights

Two models built using Random Forest Regressors and included structured feature engineering and hyperparameter tuning were compared. With the same modeling approach, the key distinction lies in the **depth of feature construction**.

### 📊 Performance Summary

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

### 💡 Business Implications
- **Profit consistently drives sales** across both models, validating a strategy that focuses on maintaining high-margin product lines.
- The [MuradAhmed00]’s model reveals that **advanced customer and discount analytics** may offer added lift — useful for fine-tuning promotional and pricing strategies.
- [nadeli11]] remains more **interpretable and operationally lightweight**, making it suitable as a baseline or in resource-constrained deployments.

=======
## Team members


    Babak S. [basaberi6]
    
    Divita Narang [divitaN-dev] Video: https://www.loom.com/share/1402e4bb1f084e5cbaaa179d16b6c161?sid=a7decfff-5e27-492a-8bca-99706fb2c72b
    
    Foram Patel [Foram2248]
    
    Murad Ahmed [MuradAhmed00] Video: https://drive.google.com/file/d/1Qzo7YNhP6uxSO0bUqlg0PiFXF7MwUxa6/view?usp=drive_link
    
    Nastaran Adeli-Rankoohi [nadeli11] 
    
    Tala Amirilarijani [Talaamiri] https://drive.google.com/file/d/1hSLFhGQFMyXxfDGa6aQ3itX59UkVD2s3/view?usp=drive_link 
