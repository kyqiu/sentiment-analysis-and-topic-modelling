import os
import pandas as pd

# Specify the absolute path to the Excel file
excel_path = r'D:\src\zweiteArtikelsentiment\annotationsentiment\swisscompare1and2and3.xlsx'

# Check if the file exists before attempting to read it
if os.path.exists(excel_path):
    # Load the Excel file into a dataframe
    df = pd.read_excel(excel_path, sheet_name='Sheet1')

    # Check if 'zonghe12' and 'zonghe13' columns exist in the dataframe
    if 'zonghe12' in df.columns and 'zonghe13' in df.columns:
        # Remove 'nan' from the strings in 'zonghe12' column and save to 'zonghe12withoutnan'
        df['zonghe12withoutnan'] = df['zonghe12'].apply(lambda x: str(x).replace('nan', ''))

        # Remove 'nan' from the strings in 'zonghe13' column and save to 'zonghe13withoutnan'
        df['zonghe13withoutnan'] = df['zonghe13'].apply(lambda x: str(x).replace('nan', ''))

        # Specify a different file path to save the updated dataframe
        updated_excel_path = r'D:\src\zweiteArtikelsentiment\annotationsentiment\swisscompare1and2and3_updated.xlsx'
        
        # Save the updated dataframe to the new Excel file
        df.to_excel(updated_excel_path, index=False)
        print(f"Dataframe successfully updated and saved to {updated_excel_path}.")
    else:
        print("Columns 'zonghe12' and 'zonghe13' not found in the dataframe.")
else:
    print(f"The file {excel_path} does not exist.")
