import pandas as pd  
import glob  

# Specify the path to the Excel folder and the file extension  
path = r'D:\src\第三篇人工智能sentiment\labelschina\totalCH\*.xlsx'  # Path to the Excel files  

# List to hold DataFrames  
dataframes = []  

# Read the Excel files  
for file in glob.glob(path):  
    df = pd.read_excel(file)  # Read each Excel file  
    dataframes.append(df)  

# Concatenate all DataFrames  
all_data = pd.concat(dataframes, ignore_index=True)  

# Debug output: Show column names and data types before conversion  
print("DataFrame columns and types before any conversion:")  
print(all_data.dtypes)  

# Ensure 'datum' column is treated as string (if it exists)  
if 'datum' in all_data.columns:  
    all_data['datum'] = all_data['datum'].astype(str)  # Convert 'datum' to string if it exists  

# Debug output: Show conversion results  
print("\nDataFrame after converting 'datum' to string:")  
print(all_data['datum'].head())  
print("\nData types after conversion:")  
print(all_data.dtypes)  

# Filter the DataFrame for entries that contain '2018', '2019', '2020', '2021' in the 'datum' column  
if 'datum' in all_data.columns:  
    filtered_data = all_data[all_data['datum'].str.contains('2022|2023')]  # Filter by presence of specified years  
else:  
    print("Warning: 'datum' column does not exist in the data.")  
    filtered_data = pd.DataFrame()  # Create an empty DataFrame if 'datum' does not exist  

# Select only the 'text' and 'datum' columns  
if 'text' in all_data.columns:  
    final_data = filtered_data[['datum', 'text']]  # Extract only the specified columns  
else:  
    print("Warning: 'text' column does not exist in the data.")  
    final_data = pd.DataFrame()  # Create an empty DataFrame if 'text' does not exist  

# Write to a new Excel file if there are valid records  
if not final_data.empty:  
    final_data.to_excel(r'D:\src\第三篇人工智能sentiment\labelschina\totalCH\combined2223CH_output.xlsx', index=False)  # Specify output path  
    print("Filtered 'text' and 'datum' columns for entries containing '2018', '2019', '2020', and '2021' written to combined_output.xlsx")  
else:  
    print("No data to write to Excel.")