import pandas as pd

# Load your CSV
df = pd.read_csv(r"D:\7th sem project\OULAD\studentInfo.csv")
print(df.head())


# Show first few rows
print(df.head())

# Show column names
print("\nColumns:\n", df.columns.tolist())

# Check if 'at_risk' column exists
print("\nContains 'at_risk' column?", 'at_risk' in df.columns)

# Check for missing values
print("\nMissing values per column:\n", df.isnull().sum())

# Show counts for categorical columns (optional)
for col in ['gender', 'region', 'disability']:
    if col in df.columns:
        print(f"\n{col} value counts:\n", df[col].value_counts())
