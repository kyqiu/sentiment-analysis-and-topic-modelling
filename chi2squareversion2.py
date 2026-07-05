import pandas as pd  
from scipy.stats import chi2_contingency  

# 读取 Excel 文件 1  
df1 = pd.read_excel(r'D:\src\第三篇人工智能sentiment\results\predicted18192021CH_output_results_one_hot.xlsx')  

# 读取 Excel 文件 2  
df2 = pd.read_excel(r'D:\src\第三篇人工智能sentiment\results\predicted2223CH_results_one_hot.xlsx')  

# 统计 "Predicted Label" 列中各标签的数量  
predicted_label_counts_1 = df1['Predicted Label'].value_counts()  
predicted_label_counts_2 = df2['Predicted Label'].value_counts()  

# 将 file 1 数据除以 2  
predicted_label_counts_1 = predicted_label_counts_1 / 2  

# 打印结果  
print("Excel File 1:")  
print("Predicted Labels 数量统计:")  
print(predicted_label_counts_1.astype(int))  

print("\nExcel File 2:")  
print("Predicted Labels 数量统计:")  
print(predicted_label_counts_2)  

# 执行卡方检验  
# 将数据转换为 2 x 5 的列联表  
contingency_table = pd.DataFrame([predicted_label_counts_1, predicted_label_counts_2]).T  
contingency_table = contingency_table.fillna(0).astype(int)  

print("\n2 x 5 列联表:")  
print(contingency_table)  

chi2, p_value, df, expected = chi2_contingency(contingency_table)  

print("\n卡方检验结果:")  
print(f"卡方统计量: {chi2:.2f}")  
print(f"p 值: {p_value:.4f}")  
print(f"自由度: {df}")  

# 根据 p 值判断是否存在显著差异  
if p_value < 0.05:  
    print("不同情感类别之间的分布存在显著差异,情感类别与时间段不独立。")  
else:  
    print("不同情感类别之间的分布不存在显著差异,情感类别与时间段是独立的。")