import pandas as pd
import os

def process_knowledge_data():
    # 读取原始数据
    if not os.path.exists("student_knowledge_mastery.csv"):
        raise FileNotFoundError("原始数据文件不存在")
    
    df = pd.read_csv("student_knowledge_mastery.csv")
    
    # 提取主知识点（不含下划线的部分）
    df['knowledge_point'] = df['knowledge_point'].apply(
        lambda x: x.split('_')[0] if '_' in x else x
    )
    
    # 计算主知识点加权平均值（按子知识点数量加权）
    result = df.groupby(['student_ID', 'knowledge_point']).agg(
        mastery_level=('mastery_level', 'mean'),
        sub_points_count=('knowledge_point', 'count')
    ).reset_index()
    
    # 保存结果
    result.to_csv("main_knowledge_mastery.csv", index=False)
    return result

# 执行处理（只需运行一次）
process_knowledge_data()