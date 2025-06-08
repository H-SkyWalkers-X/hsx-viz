import pandas as pd
import os

# 读取学生题目掌握程度数据
mastery_df = pd.read_csv('../data1/student_mastery_level.csv')

# 读取题目信息（包含知识点信息）
title_info = pd.read_csv('../data/Data_TitleInfo.csv')

# 确保数据类型一致
mastery_df['title_ID'] = mastery_df['title_ID'].astype(str)
title_info['title_ID'] = title_info['title_ID'].astype(str)

# 合并题目信息到掌握程度数据中
mastery_with_knowledge = mastery_df.merge(title_info, on='title_ID', how='left')

# 检查知识点列名
print("Columns in title_info:", title_info.columns.tolist())

# 创建结果DataFrame
knowledge_mastery_list = []

# 处理主知识点
if 'knowledge' in title_info.columns:
    main_knowledge = mastery_with_knowledge.groupby(['student_ID', 'knowledge'])['mastery_level'].mean().reset_index()
    main_knowledge['knowledge_type'] = 'main'
    main_knowledge.rename(columns={'knowledge': 'knowledge_point'}, inplace=True)
    knowledge_mastery_list.append(main_knowledge)

# 处理从属知识点
if 'sub_knowledge' in title_info.columns:
    # 过滤掉空的从属知识点
    sub_knowledge_data = mastery_with_knowledge[mastery_with_knowledge['sub_knowledge'].notna()]
    if not sub_knowledge_data.empty:
        sub_knowledge = sub_knowledge_data.groupby(['student_ID', 'sub_knowledge'])['mastery_level'].mean().reset_index()
        sub_knowledge['knowledge_type'] = 'sub'
        sub_knowledge.rename(columns={'sub_knowledge': 'knowledge_point'}, inplace=True)
        knowledge_mastery_list.append(sub_knowledge)

# 合并所有知识点掌握程度
if knowledge_mastery_list:
    final_knowledge_mastery = pd.concat(knowledge_mastery_list, ignore_index=True)
    
    # 只保留需要的列
    result_df = final_knowledge_mastery[['student_ID', 'knowledge_point', 'mastery_level']].copy()
    
    # 保存结果到CSV文件
    result_df.to_csv('../data1/student_knowledge_mastery.csv', index=False)
    
    print("计算完成，结果已保存到 '../data1/student_knowledge_mastery.csv' 文件中。")
    print(f"共处理了 {len(result_df)} 条学生-知识点记录")
    print(f"涉及 {result_df['student_ID'].nunique()} 名学生和 {result_df['knowledge_point'].nunique()} 个知识点")
    print(f"平均知识点掌握程度: {result_df['mastery_level'].mean():.3f}")
    
    # 按知识点统计
    knowledge_stats = result_df.groupby('knowledge_point')['mastery_level'].agg(['count', 'mean']).reset_index()
    knowledge_stats.columns = ['knowledge_point', 'student_count', 'avg_mastery']
    print("\n各知识点掌握情况:")
    print(knowledge_stats.sort_values('avg_mastery', ascending=False).head(10))
    
else:
    print("未找到知识点信息，请检查题目信息文件中的列名。")
