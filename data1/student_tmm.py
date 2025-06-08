import pandas as pd
import glob
import os

def generate_student_submission_details():
    """
    统计学生ID、题目ID、题目state、题目耗时、题目内存使用
    """
    # 读取所有提交记录文件
    data_folder = "../data/Data_SubmitRecord"
    csv_files = glob.glob(os.path.join(data_folder, "SubmitRecord-*.csv"))
    
    # 合并所有提交记录
    all_submissions = []
    for file in csv_files:
        df = pd.read_csv(file)
        all_submissions.append(df)
    
    # 合并所有数据
    combined_data = pd.concat(all_submissions, ignore_index=True)
    
    # 选择需要的列并重命名
    submission_details = combined_data[['student_ID', 'title_ID', 'state', 'timeconsume', 'memory']].copy()
    
    # 重命名列以使其更清晰
    submission_details.columns = ['student_ID', 'question_ID', 'submission_state', 'time_consumption', 'memory_usage']
    
    # 按学生ID和题目ID排序
    submission_details = submission_details.sort_values(['student_ID', 'question_ID'])
    
    # 保存为CSV
    output_file = "student_submission_details.csv"
    submission_details.to_csv(output_file, index=False)
    
    print(f"已生成文件: {output_file}")
    print(f"包含 {len(submission_details)} 条提交记录")
    print(f"涉及 {submission_details['student_ID'].nunique()} 名学生")
    print(f"涉及 {submission_details['question_ID'].nunique()} 道题目")
    
    # 统计各种状态的分布
    print("\n提交状态分布:")
    state_counts = submission_details['submission_state'].value_counts()
    for state, count in state_counts.items():
        percentage = (count / len(submission_details)) * 100
        print(f"{state}: {count} 次 ({percentage:.2f}%)")
    
    print("\n时间消耗统计 (秒):")
    print(f"平均耗时: {submission_details['time_consumption'].mean():.2f} 秒")
    print(f"最短耗时: {submission_details['time_consumption'].min()} 秒")
    print(f"最长耗时: {submission_details['time_consumption'].max()} 秒")
    
    print("\n内存使用统计 (KB):")
    print(f"平均内存使用: {submission_details['memory_usage'].mean():.2f} KB")
    print(f"最少内存使用: {submission_details['memory_usage'].min()} KB")
    print(f"最多内存使用: {submission_details['memory_usage'].max()} KB")
    
    print("\n前10行数据预览:")
    print(submission_details.head(10))
    
    return submission_details

def generate_aggregated_stats():
    """
    生成按学生和题目聚合的统计数据
    """
    # 读取所有提交记录文件
    data_folder = "../data/Data_SubmitRecord"
    csv_files = glob.glob(os.path.join(data_folder, "SubmitRecord-*.csv"))
    
    # 合并所有提交记录
    all_submissions = []
    for file in csv_files:
        df = pd.read_csv(file)
        all_submissions.append(df)
    
    # 合并所有数据
    combined_data = pd.concat(all_submissions, ignore_index=True)
    
    # 按学生ID和题目ID分组，计算统计数据
    aggregated_stats = combined_data.groupby(['student_ID', 'title_ID']).agg({
        'state': ['count', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]],  # 提交次数和最常见状态
        'timeconsume': ['mean', 'min', 'max'],  # 平均、最小、最大耗时
        'memory': ['mean', 'min', 'max'],  # 平均、最小、最大内存使用
        'score': ['mean', 'max']  # 平均分数和最高分数
    }).reset_index()
    
    # 重命名列
    aggregated_stats.columns = [
        'student_ID', 'question_ID', 'submission_count', 'most_common_state',
        'avg_time_consumption', 'min_time_consumption', 'max_time_consumption',
        'avg_memory_usage', 'min_memory_usage', 'max_memory_usage',
        'avg_score', 'max_score'
    ]
    
    # 四舍五入数值列
    numeric_columns = ['avg_time_consumption', 'avg_memory_usage', 'avg_score']
    for col in numeric_columns:
        aggregated_stats[col] = aggregated_stats[col].round(2)
    
    # 保存聚合统计数据
    output_file = "student_question_aggregated_stats.csv"
    aggregated_stats.to_csv(output_file, index=False)
    
    print(f"\n已生成聚合统计文件: {output_file}")
    print(f"包含 {len(aggregated_stats)} 条学生-题目组合记录")
    
    print("\n聚合统计前5行:")
    print(aggregated_stats.head())
    
    return aggregated_stats

def generate_student_performance_summary():
    """
    生成学生总体表现摘要
    """
    # 读取所有提交记录文件
    data_folder = "../data/Data_SubmitRecord"
    csv_files = glob.glob(os.path.join(data_folder, "SubmitRecord-*.csv"))
    
    # 合并所有提交记录
    all_submissions = []
    for file in csv_files:
        df = pd.read_csv(file)
        all_submissions.append(df)
    
    # 合并所有数据
    combined_data = pd.concat(all_submissions, ignore_index=True)
    
    # 按学生ID分组计算总体统计
    student_summary = combined_data.groupby('student_ID').agg({
        'title_ID': 'nunique',  # 尝试的题目数量
        'state': 'count',  # 总提交次数
        'timeconsume': 'mean',  # 平均耗时
        'memory': 'mean',  # 平均内存使用
        'score': ['mean', 'sum']  # 平均分数和总分数
    }).reset_index()
    
    # 重命名列
    student_summary.columns = [
        'student_ID', 'unique_questions_attempted', 'total_submissions',
        'avg_time_consumption', 'avg_memory_usage', 'avg_score', 'total_score'
    ]
    
    # 计算正确率
    correct_submissions = combined_data[combined_data['state'] == 'Absolutely_Correct'].groupby('student_ID').size()
    student_summary['correct_submissions'] = student_summary['student_ID'].map(correct_submissions).fillna(0)
    student_summary['accuracy_rate'] = (student_summary['correct_submissions'] / student_summary['total_submissions'] * 100).round(2)
    
    # 四舍五入数值列
    numeric_columns = ['avg_time_consumption', 'avg_memory_usage', 'avg_score', 'total_score']
    for col in numeric_columns:
        student_summary[col] = student_summary[col].round(2)
    
    # 保存学生表现摘要
    output_file = "student_performance_summary.csv"
    student_summary.to_csv(output_file, index=False)
    
    print(f"\n已生成学生表现摘要文件: {output_file}")
    print(f"包含 {len(student_summary)} 名学生的表现统计")
    
    print("\n学生表现摘要前5行:")
    print(student_summary.head())
    
    return student_summary

if __name__ == "__main__":
    print("=== 生成学生提交详细记录 ===")
    details = generate_student_submission_details()
    
    # print("\n=== 生成学生-题目聚合统计 ===")
    # aggregated = generate_aggregated_stats()
    
    # print("\n=== 生成学生表现摘要 ===")
    # summary = generate_student_performance_summary()
    
    print("\n=== 数据生成完成 ===")
