import pandas as pd

# Specify the path to your Excel file
file_path = r'D:\src\第三篇人工智能sentiment\labelschina\cleanedannotated_CHdata.xlsx'

# Read the Excel file into a DataFrame
data = pd.read_excel(file_path)

# Display the original DataFrame
print("Original DataFrame:")
print(data)

# Check and correct labels
# Assume the label column is named 'Label'. Change 'Label' to the actual column name if different.
data['label'] = data['label'].replace('hope neutral', 'neutral')

# Alternative approach if there are spaces issues or similar variations
data['label'] = data['label'].apply(lambda x: 'neutral' if 'hope neutral' in str(x) else x)

# Display the cleaned DataFrame
print("\nCleaned DataFrame (after correcting labels):")
print(data)

# Optionally, save the cleaned DataFrame back to an Excel file
data.to_excel('cleanedannotated_CHdata.xlsx', index=False)