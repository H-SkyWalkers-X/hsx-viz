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

# 将 timeconsume 转换为数值类型，无法转换的值设置为 NaN
correct_records = correct_records.copy()
correct_records['timeconsume'] = pd.to_numeric(correct_records['timeconsume'], errors='coerce')

# 删除 timeconsume 为 NaN 的记录
correct_records = correct_records.dropna(subset=['timeconsume'])

# 选择每个学生在每个题目上的最少耗时
student_min_times = correct_records.groupby(['student_ID', 'title_ID'])['timeconsume'].min().reset_index()
student_min_times.rename(columns={'timeconsume': 'min_student_time'}, inplace=True)

# 计算每个题目上所有学生最少时间的最小值和最大值
title_time_stats = student_min_times.groupby('title_ID')['min_student_time'].agg(['min', 'max']).reset_index()
title_time_stats.rename(columns={'min': 'title_min_time', 'max': 'title_max_time'}, inplace=True)

# 合并统计信息到学生最少时间记录中
result = student_min_times.merge(title_time_stats, on='title_ID', how='left')

# 计算归一化时间效率
def calculate_time_efficiency(row):
    min_time = row['title_min_time']
    max_time = row['title_max_time']
    student_time = row['min_student_time']
    
    if max_time == min_time:
        return 1.0  # 如果最大最小时间相等，效率为1
    
    # 归一化：(max_time - student_time) / (max_time - min_time)
    # 这样最短时间的学生效率为1，最长时间的学生效率为0
    efficiency = (max_time - student_time) / (max_time - min_time)
    return efficiency

result['time_efficiency'] = result.apply(calculate_time_efficiency, axis=1)

# 添加是否为该题目最短时间的标记
result['is_fastest'] = result['min_student_time'] == result['title_min_time']

# 去除重复记录
result = result.drop_duplicates()

# 避免保存重复的记录
final_result = result.drop_duplicates(subset=['student_ID', 'title_ID'], keep='first')
# print(final_result)
# 保存结果到CSV文件
final_result.to_csv('../data1/student_time_efficiency.csv', index=False)

print("计算完成，结果已保存到 '../data1/student_time_efficiency.csv' 文件中。")
print(f"共处理了 {len(result)} 条学生-题目记录")
print(f"涉及 {result['student_ID'].nunique()} 名学生和 {result['title_ID'].nunique()} 道题目")