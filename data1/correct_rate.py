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

# 确保数据类型一致
all_submit_records['title_ID'] = all_submit_records['title_ID'].astype(str)
all_submit_records['student_ID'] = all_submit_records['student_ID'].astype(str)

# 计算每个学生在每个题目上的总提交次数和正确次数
submit_stats = all_submit_records.groupby(['student_ID', 'title_ID']).agg({
    'state': [
        'count',  # 总提交次数
        lambda x: (x == 'Absolutely_Correct').sum()  # 正确提交次数
    ]
}).reset_index()

# 展平列名
submit_stats.columns = ['student_ID', 'title_ID', 'total_submissions', 'correct_submissions']

# 计算正确率 = 正确提交次数 / 总提交次数
submit_stats['correct_rate'] = submit_stats['correct_submissions'] / submit_stats['total_submissions']

# 添加是否完全正确的标记（正确率为1）
submit_stats['is_perfect'] = submit_stats['correct_rate'] == 1.0

# 添加是否有正确提交的标记
submit_stats['has_correct'] = submit_stats['correct_submissions'] > 0

# 保存结果到CSV文件
submit_stats.to_csv('../data1/student_correct_rate.csv', index=False)

print("计算完成，结果已保存到 '../data1/student_correct_rate.csv' 文件中。")
print(f"共处理了 {len(submit_stats)} 条学生-题目记录")
print(f"涉及 {submit_stats['student_ID'].nunique()} 名学生和 {submit_stats['title_ID'].nunique()} 道题目")
print(f"平均正确率: {submit_stats['correct_rate'].mean():.3f}")
print(f"完全正确的记录数: {submit_stats['is_perfect'].sum()}")
