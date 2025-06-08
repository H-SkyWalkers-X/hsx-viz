import pandas as pd
import glob
import os
from datetime import datetime
import numpy as np

def calculate_daily_study_time(df):
    """
    计算每个学生每天的答题时长
    只考虑时间间隔小于30分钟（1800秒）的记录
    """
    daily_study_time = {}
    
    for student_id in df['student_ID'].unique():
        student_data = df[df['student_ID'] == student_id].copy()
        
        # 将时间戳转换为datetime并按时间排序
        student_data['datetime'] = pd.to_datetime(student_data['time'], unit='s')
        student_data = student_data.sort_values('datetime')
        
        # 按日期分组
        student_data['date'] = student_data['datetime'].dt.date
        
        for date in student_data['date'].unique():
            date_data = student_data[student_data['date'] == date].copy()
            date_data = date_data.sort_values('datetime')
            
            total_study_time = 0
            
            if len(date_data) > 1:
                for i in range(1, len(date_data)):
                    time_diff = (date_data.iloc[i]['datetime'] - date_data.iloc[i-1]['datetime']).total_seconds()
                    
                    # 只考虑时间间隔小于30分钟（1800秒）的记录
                    if time_diff <= 1800:
                        total_study_time += time_diff
            
            daily_study_time[(student_id, date)] = total_study_time / 60  # 转换为分钟
    
    return daily_study_time

def generate_daily_student_stats():
    """
    生成学生每日学习统计数据
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
    
    # 添加日期时间列
    combined_data['datetime'] = pd.to_datetime(combined_data['time'], unit='s')
    combined_data['date'] = combined_data['datetime'].dt.date
    
    # 计算每日答题时长
    print("正在计算每日答题时长...")
    daily_study_time = calculate_daily_study_time(combined_data)
    
    # 计算每日提交次数和平均得分
    print("正在计算每日提交次数和平均得分...")
    daily_stats = combined_data.groupby(['student_ID', 'date']).agg({
        'score': ['count', 'mean']  # 提交次数和平均得分
    }).reset_index()
    
    # 重命名列
    daily_stats.columns = ['student_ID', 'date', 'daily_submissions', 'avg_score']
    
    # 添加每日答题时长
    daily_stats['daily_study_time'] = daily_stats.apply(
        lambda row: daily_study_time.get((row['student_ID'], row['date']), 0),
        axis=1
    )
    
    # 重新排列列的顺序
    daily_stats = daily_stats[['student_ID', 'date', 'daily_study_time', 'daily_submissions', 'avg_score']]
    
    # 四舍五入到合理的精度
    daily_stats['daily_study_time'] = daily_stats['daily_study_time'].round(2)
    daily_stats['avg_score'] = daily_stats['avg_score'].round(3)
    
    # 将日期转换为字符串格式
    daily_stats['date'] = daily_stats['date'].astype(str)
    
    # 保存为CSV
    output_file = "student_daily_stats.csv"
    daily_stats.to_csv(output_file, index=False)
    
    print(f"已生成文件: {output_file}")
    print(f"包含 {len(daily_stats)} 条每日学习记录")
    print(f"涉及 {daily_stats['student_ID'].nunique()} 名学生")
    print(f"时间范围: {daily_stats['date'].min()} 到 {daily_stats['date'].max()}")
    
    print("\n数据统计摘要:")
    print(f"平均每日答题时长: {daily_stats['daily_study_time'].mean():.2f} 分钟")
    print(f"平均每日提交次数: {daily_stats['daily_submissions'].mean():.2f} 次")
    print(f"平均每次提交得分: {daily_stats['avg_score'].mean():.3f} 分")
    
    print("\n前10行数据预览:")
    print(daily_stats.head(10))
    
    return daily_stats

def generate_student_summary():
    """
    生成学生总体统计摘要
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
    
    # 添加日期时间列
    combined_data['datetime'] = pd.to_datetime(combined_data['time'], unit='s')
    combined_data['date'] = combined_data['datetime'].dt.date
    
    # 计算每日答题时长
    daily_study_time = calculate_daily_study_time(combined_data)
    
    # 按学生汇总统计
    student_summary = []
    
    for student_id in combined_data['student_ID'].unique():
        student_data = combined_data[combined_data['student_ID'] == student_id]
        
        # 计算总答题时长
        total_study_time = sum([time for (sid, date), time in daily_study_time.items() if sid == student_id])
        
        # 计算活跃天数
        active_days = student_data['date'].nunique()
        
        # 计算总提交次数
        total_submissions = len(student_data)
        
        # 计算平均得分
        avg_score = student_data['score'].mean()
        
        student_summary.append({
            'student_ID': student_id,
            'total_study_time': round(total_study_time, 2),
            'active_days': active_days,
            'total_submissions': total_submissions,
            'avg_score': round(avg_score, 3),
            'avg_daily_study_time': round(total_study_time / active_days if active_days > 0 else 0, 2),
            'avg_daily_submissions': round(total_submissions / active_days if active_days > 0 else 0, 2)
        })
    
    summary_df = pd.DataFrame(student_summary)
    
    # 保存学生总体统计
    output_file = "student_summary_stats.csv"
    summary_df.to_csv(output_file, index=False)
    
    print(f"\n已生成学生总体统计文件: {output_file}")
    print(f"包含 {len(summary_df)} 名学生的总体统计")
    
    print("\n学生总体统计前5行:")
    print(summary_df.head())
    
    return summary_df

if __name__ == "__main__":
    print("=== 生成学生每日学习统计 ===")
    daily_stats = generate_daily_student_stats()
    
    print("\n=== 生成学生总体统计摘要 ===")
    summary_stats = generate_student_summary()
    
    print("\n=== 数据生成完成 ===")
