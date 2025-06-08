import pandas as pd
import os

# 加载数据
student_info_path = '../data/Data_StudentInfo.csv'
title_info_path = '../data/Data_TitleInfo.csv'

student_info = pd.read_csv(student_info_path)
title_info = pd.read_csv(title_info_path)

# 初始化一个空的DataFrame用于存储所有答题记录
all_submit_records = pd.DataFrame()

# 遍历15个班级的答题日志文件
for i in range(1, 16):
    file_path = f'../data/Data_SubmitRecord/SubmitRecord-Class{i}.csv'
    if os.path.exists(file_path):
        submit_records = pd.read_csv(file_path)
        all_submit_records = pd.concat([all_submit_records, submit_records])

# 检查列名
print("Columns in all_submit_records:", all_submit_records.columns)
print("Columns in title_info:", title_info.columns)

# 确保数据类型一致
all_submit_records['title_ID'] = all_submit_records['title_ID'].astype(str)
title_info['title_ID'] = title_info['title_ID'].astype(str)

# 合并题目信息到答题记录中
all_submit_records = all_submit_records.merge(title_info, on='title_ID', how='left', suffixes=('', '_title'))

# 筛选出 state 为 Absolutely_Correct 的记录
correct_records = all_submit_records[all_submit_records['state'] == 'Absolutely_Correct']

# 将 memory 转换为数值类型，无法转换的值设置为 NaN
correct_records = correct_records.copy()
correct_records['memory'] = pd.to_numeric(correct_records['memory'], errors='coerce')

# 删除 memory 为 NaN 的记录
correct_records = correct_records.dropna(subset=['memory'])

# 选择每个学生在每个题目上的最少内存消耗
student_min_memory = correct_records.groupby(['student_ID', 'title_ID'])['memory'].min().reset_index()
student_min_memory.rename(columns={'memory': 'min_student_memory'}, inplace=True)

# 计算每个题目上所有学生最少内存的最小值和最大值
title_memory_stats = student_min_memory.groupby('title_ID')['min_student_memory'].agg(['min', 'max']).reset_index()
title_memory_stats.rename(columns={'min': 'title_min_memory', 'max': 'title_max_memory'}, inplace=True)

# 合并统计信息到学生最少内存记录中
result = student_min_memory.merge(title_memory_stats, on='title_ID', how='left')

# 计算归一化内存效率
def calculate_memory_efficiency(row):
    min_memory = row['title_min_memory']
    max_memory = row['title_max_memory']
    student_memory = row['min_student_memory']
    
    if max_memory == min_memory:
        return 1.0  # 如果最大最小内存相等，效率为1
    
    # 归一化：(max_memory - student_memory) / (max_memory - min_memory)
    # 这样最少内存的学生效率为1，最多内存的学生效率为0
    efficiency = (max_memory - student_memory) / (max_memory - min_memory)
    return efficiency

result['memory_efficiency'] = result.apply(calculate_memory_efficiency, axis=1)

# 添加是否为该题目最少内存的标记
result['is_lowest_memory'] = result['min_student_memory'] == result['title_min_memory']

# 去除重复记录
result = result.drop_duplicates()

# 避免保存重复的记录
final_result = result.drop_duplicates(subset=['student_ID', 'title_ID'], keep='first')

# 保存结果到CSV文件
final_result.to_csv('../data1/student_memory_efficiency.csv', index=False)

print("计算完成，结果已保存到 '../data1/student_memory_efficiency.csv' 文件中。")
print(f"共处理了 {len(result)} 条学生-题目记录")
print(f"涉及 {result['student_ID'].nunique()} 名学生和 {result['title_ID'].nunique()} 道题目")