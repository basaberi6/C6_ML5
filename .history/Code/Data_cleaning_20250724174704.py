# data_cleaning.py

import pandas as pd

# Step 1: Load the dataset
file_path = "../DSI_Project/C6_ML5/Data/Raw/Sample - Superstore.csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Step 2: Preview data
print("Dataset shape:", df.shape)
print("First 5 rows:")
print(df.head())

# Step 3: Convert data types
# Convert dates to datetime objects
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df['Ship_Date'] = pd.to_datetime(df['Ship_Date'])

# Optional: Convert string columns with low cardinality to 'category' type
category_columns = ['Ship_Mode', 'Segment', 'Region', 'Category', 'Sub-Category']
for col in category_columns:
    df[col] = df[col].astype('category')

# Step 4: Handle missing values
# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Drop rows with missing 'Region' values
df = df.dropna(subset=['Region'])

# Step 5: Remove duplicate rows (if any)
df = df.drop_duplicates()

# Step 6: Save cleaned dataset
cleaned_path = "../DSI_Project/C6_ML5/Data/Processed/cleaned_superstore.csv"
df.to_csv(cleaned_path, index=False)

print("\n✅ Data cleaning complete. Cleaned file saved at:")
print(cleaned_path)
