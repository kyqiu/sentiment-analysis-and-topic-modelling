


import pandas as pd

# assume 'df' is your Pandas DataFrame
df = pd.read_excel(r"D:\src\第三篇人工智能sentiment\sentimentproject\cleanedannotated_Swissdata.xlsx")
# Method 1: Using df.info()
print(df.info())

# Method 2: Using df.dtypes
print(df.dtypes)

# Method 3: Using pd.DataFrame.select_dtypes()
print(df.select_dtypes(include=[object]))  # show only object (string) columns
print(df.select_dtypes(exclude=[object]))  # show all non-object (numeric) columns

# Method 4: Iterate over columns and print data types
for col in df.columns:
    print(f"Column {col}: {df[col].dtype}")