import pandas as pd
import os

# 读取各个效率指标的CSV文件
correct_rate_df = pd.read_csv('../data1/student_correct_rate.csv')
time_efficiency_df = pd.read_csv('../data1/student_time_efficiency.csv')
memory_efficiency_df = pd.read_csv('../data1/student_memory_efficiency.csv')

# 确保数据类型一致
for df in [correct_rate_df, time_efficiency_df, memory_efficiency_df]:
    df['student_ID'] = df['student_ID'].astype(str)
    df['title_ID'] = df['title_ID'].astype(str)

# 合并所有效率指标
master_df = correct_rate_df[['student_ID', 'title_ID', 'correct_rate']].copy()

# 合并时间效率
master_df = master_df.merge(
    time_efficiency_df[['student_ID', 'title_ID', 'time_efficiency']], 
    on=['student_ID', 'title_ID'], 
    how='left'
)

# 合并内存效率
master_df = master_df.merge(
    memory_efficiency_df[['student_ID', 'title_ID', 'memory_efficiency']], 
    on=['student_ID', 'title_ID'], 
    how='left'
)

# 填充缺失值为0（表示该学生在该题目上没有正确提交记录）
master_df['time_efficiency'] = master_df['time_efficiency'].fillna(0)
master_df['memory_efficiency'] = master_df['memory_efficiency'].fillna(0)

# 设置权重（可根据需要调整）
weights = {
    'correct_rate': 0.4,        # 正确率权重40%
    'time_efficiency': 0.3,     # 时间效率权重30%
    'memory_efficiency': 0.2,   # 内存效率权重20%
    'score_rate': 0.1          # 得分率权重10%（暂时设为0，因为没有得分数据）
}

# 计算加权掌握程度
master_df['mastery_level'] = (
    master_df['correct_rate'] * weights['correct_rate'] +
    master_df['time_efficiency'] * weights['time_efficiency'] +
    master_df['memory_efficiency'] * weights['memory_efficiency']
    # + master_df['score_rate'] * weights['score_rate']  # 如果有得分数据可以添加
)

# 只保留需要的列
result_df = master_df[['student_ID', 'title_ID', 'mastery_level']].copy()

# 保存结果到CSV文件
result_df.to_csv('../data1/student_mastery_level.csv', index=False)

print("计算完成，结果已保存到 '../data1/student_mastery_level.csv' 文件中。")
print(f"共处理了 {len(result_df)} 条学生-题目记录")
print(f"涉及 {result_df['student_ID'].nunique()} 名学生和 {result_df['title_ID'].nunique()} 道题目")
print(f"平均掌握程度: {result_df['mastery_level'].mean():.3f}")
print(f"掌握程度分布:")
print(f"  0.8以上: {(result_df['mastery_level'] >= 0.8).sum()} 条记录")
print(f"  0.6-0.8: {((result_df['mastery_level'] >= 0.6) & (result_df['mastery_level'] < 0.8)).sum()} 条记录")
print(f"  0.4-0.6: {((result_df['mastery_level'] >= 0.4) & (result_df['mastery_level'] < 0.6)).sum()} 条记录")
print(f"  0.4以下: {(result_df['mastery_level'] < 0.4).sum()} 条记录")
