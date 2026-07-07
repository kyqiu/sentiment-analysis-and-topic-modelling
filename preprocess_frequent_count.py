import pandas as pd
from collections import Counter

# 读取 Excel 文件
df = pd.read_excel(r"D:\src\zweiteArtikelsentiment\annotationsentiment\chinacompare1and2and3_preprocessed.xlsx")

# 获取指定列的数据
data_column = df.iloc[:, 4:25].values.flatten()

# 统计字符串频率
string_counter = Counter(data_column)

# 将结果按频率从大到小排序
sorted_strings = sorted(string_counter.items(), key=lambda x: x[1], reverse=True)

# 创建一个新的 DataFrame 以保存结果
result_df = pd.DataFrame(sorted_strings, columns=['String', 'Frequency'])

# 将结果保存到新的 Excel 文件
result_df.to_excel(r"D:\src\zweiteArtikelsentiment\annotationsentiment\chinacompare1and2and3string_frequency.xlsx", index=False)
