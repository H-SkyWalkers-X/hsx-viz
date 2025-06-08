import pandas as pd
import glob
import os

def extract_student_time_data():
    """
    从SubmitRecord-Class{i}.csv文件中提取学生ID和编程时间数据
    """
    # 获取当前目录下所有匹配的CSV文件
    csv_files = glob.glob("../data/Data_SubmitRecord/SubmitRecord-Class*.csv")
    
    if not csv_files:
        print("未找到匹配的CSV文件")
        return None
    
    all_data = []
    
    for file in csv_files:
        try:
            # 读取CSV文件
            df = pd.read_csv(file)
            
            # 提取班级编号
            class_num = file.split('Class')[1].split('.csv')[0]
            
            # 提取学生ID和编程时间消耗
            if 'student_ID' in df.columns and 'time' in df.columns:
                student_data = df[['student_ID', 'time']].copy()
                
                all_data.append(student_data)
            else:
                print(f"文件 {file} 中未找到所需列")
                
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
    
    if all_data:
        # 合并所有数据
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # 保存为新的CSV文件
        output_file = "student_time_cost_summary.csv"
        combined_data.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"数据已保存到 {output_file}")
        
        return combined_data
    else:
        print("没有成功提取任何数据")
        return None

if __name__ == "__main__":
    result = extract_student_time_data()
    if result is not None:
        print("\n数据预览:")
        print(result.head())
