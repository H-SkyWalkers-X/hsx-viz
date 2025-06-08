import pandas as pd
import glob
import os

def generate_student_language_usage():
    """
    分析学生使用编程语言次数在提交次数中的占比
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
    
    # 编程语言映射（根据method字段推断）
    language_mapping = {
        'Method_5Q4KoXthUuYz3bvrTDFm': 'Language1',
        'Method_gj1NLb4Jn7URf9K2kQPd': 'Language2',
        'Method_BXr9AIsPQhwNvyGdZL57': 'Language3',
        'Method_Cj9Ya2R7fZd6xs1q5mNQ': 'Language4',
        'Method_m8vwGkEZc3TSW2xqYUoR': 'Language5',
    }
    
    # 添加编程语言列
    combined_data['language'] = combined_data['method'].map(language_mapping)
    
    # 按学生ID分组统计
    student_stats = []
    
    for student_id in combined_data['student_ID'].unique():
        student_data = combined_data[combined_data['student_ID'] == student_id]
        total_submissions = len(student_data)
        
        # 统计各语言使用次数和占比
        language_counts = student_data['language'].value_counts()
        language_ratios = student_data['language'].value_counts(normalize=True)
        
        # 创建学生记录
        student_record = {
            'student_ID': student_id,
            'total_submissions': total_submissions
        }
        
        # 添加各语言的使用次数和占比
        for lang in ['Language1', 'Language2', 'Language3', 'Language4', 'Language5']:
            student_record[f'{lang}_count'] = language_counts.get(lang, 0)
            student_record[f'{lang}_ratio'] = round(language_ratios.get(lang, 0), 3)
        
        student_stats.append(student_record)
    
    # 转换为DataFrame
    result_df = pd.DataFrame(student_stats)
    
    # 保存为CSV
    output_file = "student_language_usage.csv"
    result_df.to_csv(output_file, index=False)
    
    print(f"已生成文件: {output_file}")
    print(f"包含 {len(result_df)} 名学生的编程语言使用统计")
    print("\n前5行数据预览:")
    print(result_df.head())
    
    return result_df

def generate_simplified_version():
    """
    生成简化版本，只包含主要使用的编程语言占比
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
    
    # 编程语言映射
    language_mapping = {
        'Method_5Q4KoXthUuYz3bvrTDFm': 'Language1',
        'Method_gj1NLb4Jn7URf9K2kQPd': 'Language2', 
        'Method_BXr9AIsPQhwNvyGdZL57': 'Language3',
        'Method_Cj9Ya2R7fZd6xs1q5mNQ': 'Language4',
        'Method_m8vwGkEZc3TSW2xqYUoR': 'Language5'
    }
    
    # 添加编程语言列
    combined_data['language'] = combined_data['method'].map(language_mapping)
    
    # 按学生ID分组，计算每种语言的使用占比
    student_language_ratios = []
    
    for student_id in combined_data['student_ID'].unique():
        student_data = combined_data[combined_data['student_ID'] == student_id]
        language_ratios = student_data['language'].value_counts(normalize=True)
        
        # 为每种语言创建一条记录
        for language, ratio in language_ratios.items():
            student_language_ratios.append({
                'student_ID': student_id,
                'knowledge_point': language,
                'mastery_level': round(ratio, 3)
            })
    
    # 转换为DataFrame
    result_df = pd.DataFrame(student_language_ratios)
    
    # 保存为CSV
    output_file = "student_language_usage2.csv"
    result_df.to_csv(output_file, index=False)
    
    print(f"已生成简化版文件: {output_file}")
    print(f"包含 {len(result_df)} 条记录")
    print("\n前10行数据预览:")
    print(result_df.head(10))
    
    return result_df

if __name__ == "__main__":
    print("=== 生成详细版本 ===")
    detailed_df = generate_student_language_usage()
    
    print("\n=== 生成简化版本（适用于可视化） ===")
    # simplified_df = generate_simplified_version()


