import pandas as pd
import os

# 加载数据
student_info = pd.read_csv('../data/Data_StudentInfo.csv')
title_info = pd.read_csv('../data/Data_TitleInfo.csv')


# 初始化一个空的DataFrame用于存储所有答题记录
all_submit_records = pd.DataFrame()

# 遍历15个班级的答题日志文件
for i in range(1, 16):
    file_path = f'../data/Data_SubmitRecord/SubmitRecord-Class{i}.csv'
    if os.path.exists(file_path):
        submit_records = pd.read_csv(file_path)
        all_submit_records = pd.concat([all_submit_records, submit_records])

# 合并题目信息到答题记录中
all_submit_records = all_submit_records.merge(title_info, on='title_ID', how='left', suffixes=('', '_title'))

# 检查合并后的列名
print(all_submit_records.columns)

# 计算每个学习者在每个题目的答题得分率
def calculate_score_rate(group):
    total_score = group['score'].sum()  # 学习者在该题目的总得分
    total_possible_score = group['score_title'].iloc[0] * len(group)  # 题目总分 * 提交次数
    return total_score / total_possible_score if total_possible_score > 0 else 0

# 按学习者和题目分组，并计算得分率
result = all_submit_records.groupby(['student_ID', 'title_ID']).apply(calculate_score_rate).reset_index(name='score_rate')

# 保存结果到CSV文件
result.to_csv('./student_score_rate.csv', index=False)
print(result)
print("计算完成，结果已保存到 'score_rate.csv' 文件中。")

'''rate为0的一般认定为没有刷题或者没有得分，所以认定为没有掌握这个题目'''