import pandas as pd
import glob
import os
from datetime import datetime

def process_submit_records():
    # 获取所有CSV文件路径
    csv_files = glob.glob('./data/Data_SubmitRecord/*.csv')
    
    if not csv_files:
        print("未找到CSV文件")
        return
    
    all_data = []
    
    # 处理每个CSV文件
    for file_path in csv_files:
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 检查必要的列是否存在
            if 'time' in df.columns and 'student_ID' in df.columns:
                # 提取需要的列
                selected_data = df[['time', 'student_ID']].copy()
                
                # 将时间戳转换为日期时间格式
                selected_data['time'] = pd.to_datetime(selected_data['time'], unit='s')
                
                all_data.append(selected_data)
                print(f"已处理文件: {os.path.basename(file_path)}")
            else:
                print(f"文件 {os.path.basename(file_path)} 缺少必要的列")
                
        except Exception as e:
            print(f"处理文件 {os.path.basename(file_path)} 时出错: {e}")
    
    # 合并所有数据
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # 保存到新的CSV文件
        output_file = './data/processed_submit_records.csv'
        combined_data.to_csv(output_file, index=False)
        print(f"数据已保存到: {output_file}")
        print(f"总共处理了 {len(combined_data)} 条记录")
    else:
        print("没有找到有效的数据")

if __name__ == "__main__":
    process_submit_records()
