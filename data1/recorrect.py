import pandas as pd
import os
import glob

def process_csv_files():
    """
    处理当前目录下所有CSV文件，检查并修正学生学号长度
    """
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 查找所有CSV文件
    csv_files = glob.glob(os.path.join(current_dir, "*.csv"))
    
    if not csv_files:
        print("当前目录下没有找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        print(f"\n处理文件: {filename}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 检查是否有student_ID列
            if 'student_ID' not in df.columns:
                print(f"  跳过: 文件中没有'student_ID'列")
                continue
            
            # 检查学号长度并修正
            original_count = len(df)
            modified_count = 0
            
            # 转换为字符串类型处理
            df['student_ID'] = df['student_ID'].astype(str)
            
            # 记录修改前的状态
            before_21_count = len(df[df['student_ID'].str.len() == 21])
            before_20_count = len(df[df['student_ID'].str.len() == 20])
            
            # 如果学号是21位，去掉最后一位
            mask_21_digits = df['student_ID'].str.len() == 21
            if mask_21_digits.any():
                df.loc[mask_21_digits, 'student_ID'] = df.loc[mask_21_digits, 'student_ID'].str[:-1]
                modified_count = mask_21_digits.sum()
            
            # 记录修改后的状态
            after_21_count = len(df[df['student_ID'].str.len() == 21])
            after_20_count = len(df[df['student_ID'].str.len() == 20])
            
            print(f"  修改前: 20位学号 {before_20_count} 个, 21位学号 {before_21_count} 个")
            print(f"  修改后: 20位学号 {after_20_count} 个, 21位学号 {after_21_count} 个")
            print(f"  共修正了 {modified_count} 个学号")
            
            # 如果有修改，保存文件
            if modified_count > 0:
                df.to_csv(csv_file, index=False)
                print(f"  ✓ 文件已更新保存")
            else:
                print(f"  ✓ 文件无需修改")
                
        except Exception as e:
            print(f"  ✗ 处理文件时出错: {str(e)}")
    
    print("\n处理完成！")

if __name__ == "__main__":
    process_csv_files()
