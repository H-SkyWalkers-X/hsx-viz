import streamlit as st
import numpy as np
import pandas as pd
from streamlit_echarts import st_echarts
from streamlit_echarts import st_pyecharts
from pyecharts import options as opts
from pyecharts.charts import Scatter,Graph,Pie,Polar,Bar,Grid,Calendar,Radar
import os
import random
import datetime
import time
from sklearn.decomposition import PCA
import plotly.express as px




# 设置页面布局为宽屏模式
st.set_page_config(
    layout="wide",
    page_title="「析数启智」时序多变量教育数据可视分析",
    initial_sidebar_state="collapsed"
)

# 创建示例数据
data = np.random.randn(20, 3)
df = pd.DataFrame(
    data,
    columns=['类别A', '类别B', '类别C']
)

# 初始化会话状态
if 'expanded_nodes' not in st.session_state:
    st.session_state.expanded_nodes = set()
if 'clicked_data' not in st.session_state:
    st.session_state.clicked_data = None
if 'click_history' not in st.session_state:
    st.session_state.click_history = []
if 'last_processed_click' not in st.session_state:
    st.session_state.last_processed_click = None
if 'click_timestamp' not in st.session_state:
    st.session_state.click_timestamp = 0
# 新增：为散点图点击添加独立的状态管理
if 'selected_title_id' not in st.session_state:
    st.session_state.selected_title_id = 'Question_3MwAFlmNO8EKrpY5zjUd'
if 'selected_question_id' not in st.session_state:
    st.session_state.selected_question_id = None
# 新增：固定随机数据，避免每次重新生成
if 'polar_data' not in st.session_state:
    st.session_state.polar_data = [(i, random.randint(1, 100)) for i in range(10)]
if 'calendar_data' not in st.session_state:
    begin = datetime.date(2024, 9, 1)
    end = datetime.date(2025, 1, 1)
    st.session_state.calendar_data = [
        [str(begin + datetime.timedelta(days=i)), random.randint(1000, 25000)]
        for i in range((end - begin).days + 1)
    ]

#! 所有函数定义
def load_data():
    student_info = pd.read_csv('./data/Data_StudentInfo.csv')
    title_info = pd.read_csv('./data/Data_TitleInfo.csv')
    knowledge_mastery = pd.read_csv('./data1/student_knowledge_mastery.csv')
    submission_details=pd.read_csv('./data1/student_submission_details.csv')
    language_usage=pd.read_csv("./data1/student_language_usage.csv")
    main_knowledge_mastery = pd.read_csv("./data1/main_knowledge_mastery.csv")
    score_rate_data = pd.read_csv('./data1/student_score_rate.csv')
    student_daily_stats = pd.read_csv("./data1/student_daily_stats.csv")
    student_daily_stats['date'] = pd.to_datetime(student_daily_stats['date']).dt.date
    all_submit_records = pd.DataFrame()

    for i in range(1, 16):
        file_path = f'./data/Data_SubmitRecord/SubmitRecord-Class{i}.csv'
        if os.path.exists(file_path):
            submit_records = pd.read_csv(file_path)
            all_submit_records = pd.concat([all_submit_records, submit_records])

    return student_info, title_info, knowledge_mastery, all_submit_records,submission_details,language_usage,main_knowledge_mastery,score_rate_data,student_daily_stats

# 预处理数据
def preprocess_data(student_info, title_info, knowledge_mastery, all_submit_records):
    # 筛选出 state 为 Absolutely_Correct 的记录
    correct_records = all_submit_records[all_submit_records['state'] == 'Absolutely_Correct']
    # 合并题目信息和学生信息到答题记录中
    correct_records = correct_records.merge(title_info, on='title_ID', how='left')
    correct_records = correct_records.merge(student_info, on='student_ID', how='left')
    # 去除重复记录和无效数据
    correct_records.dropna(inplace=True)
    correct_records.drop_duplicates(inplace=True)
    return correct_records

@st.cache_data
def preprocess_data2(student_info, title_info, knowledge_mastery, all_submit_records):
    # 标准化列名（去除前后空格）
    all_submit_records.columns = all_submit_records.columns.str.strip()
    title_info.columns = title_info.columns.str.strip()
    student_info.columns = student_info.columns.str.strip()
    
    # 合并题目信息到提交记录
    records_with_info = all_submit_records.merge(
        title_info[['title_ID', 'knowledge', 'sub_knowledge']], 
        on='title_ID', 
        how='left'
    )
    
    # 合并学生信息
    records_with_info = records_with_info.merge(
        student_info[['student_ID', 'sex', 'age', 'major']],
        on='student_ID',
        how='left'
    )
    
    # 去除无效数据
    records_with_info.dropna(subset=['knowledge', 'sub_knowledge'], inplace=True)
    records_with_info.drop_duplicates(inplace=True)
    
    return records_with_info

# 3. 构建四层桑基图数据
def build_sankey_data(student_data, selected_student):
    # 四层：学生 → 知识点 → 子知识点 → 题目
    nodes = []
    links = []
    
    # 第一层：学生节点（只有一个大的）
    student_node = f"学生 {selected_student}"
    nodes.append({
        "name": student_node,
        "itemStyle": {"color": "#8B0000"}  # 深红色 - 学生节点
    })
    
    # 第二层：知识点（总是显示）
    knowledge_points = student_data['knowledge'].unique()
    for knowledge in knowledge_points:
        nodes.append({
            "name": knowledge,
            "itemStyle": {"color": "#1f77b4"}  # 蓝色 - 知识点
        })
        
        # 学生 → 知识点的连接
        knowledge_data = student_data[student_data['knowledge'] == knowledge]
        value = len(knowledge_data)
        links.append({
            "source": student_node,
            "target": knowledge,
            "value": value
        })
        
        # 第三层：子知识点（总是显示）
        sub_knowledge_points = knowledge_data['sub_knowledge'].unique()
        for sub_knowledge in sub_knowledge_points:
            if not any(n["name"] == sub_knowledge for n in nodes):
                nodes.append({
                    "name": sub_knowledge,
                    "itemStyle": {"color": "#ff7f0e"}  # 橙色 - 子知识点
                })
            
            # 知识点 → 子知识点的连接
            sub_data = knowledge_data[knowledge_data['sub_knowledge'] == sub_knowledge]
            value = len(sub_data)
            links.append({
                "source": knowledge,
                "target": sub_knowledge,
                "value": value
            })
            
            # 第四层：题目（只有展开的子知识点才显示）
            if sub_knowledge in st.session_state.expanded_nodes:
                title_ids = sub_data['title_ID'].unique()
                for title_id in title_ids:
                    # 使用原始题目ID作为显示名称
                    title_name = str(title_id)
                    if not any(n["name"] == title_name for n in nodes):
                        nodes.append({
                            "name": title_name,
                            "itemStyle": {"color": "#4dad4d"}  # 绿色 - 题目
                        })
                    
                    # 子知识点 → 题目的连接
                    title_data = sub_data[sub_data['title_ID'] == title_id]
                    value = len(title_data)
                    links.append({
                        "source": sub_knowledge,
                        "target": title_name,
                        "value": value
                    })
    
    return nodes, links

# 4. 创建桑基图配置
def create_sankey_options(student_data, selected_student):
    nodes, links = build_sankey_data(student_data, selected_student)
    
    options = {
        "title": {
            "text": f"学生-知识点-题目桑基图",
            "left": "left",
            
            'textStyle':{'color':"#000000",
                         'fontSize':25}
        },
        "tooltip": {
            "trigger": "item",
            "triggerOn": "mousemove"
        },
        "series": [
            {
                "type": "sankey",
                "data": nodes,
                "links": links,
                "emphasis": {
                    "focus": "adjacency"
                },
                "lineStyle": {
                    "color": "gradient",
                    "curveness": 0.5
                },
                "label": {
                    "position": "right"
                },
                "nodeGap": 20,
                "nodeWidth": 20,
                "layoutIterations": 32
            }
        ]
    }
    
    return options

# 5. 判断节点层级
def get_node_level(node_name, student_data, selected_student):
    """判断节点属于哪一层"""
    # 第一层：学生节点
    if node_name == f"学生 {selected_student}":
        return 1
    
    # 第二层：知识点
    if node_name in student_data['knowledge'].unique():
        return 2
    
    # 第三层：子知识点
    if node_name in student_data['sub_knowledge'].unique():
        return 3
    
    # 第四层：题目 - 检查是否为题目ID
    if str(node_name) in student_data['title_ID'].astype(str).unique():
        return 4
    
    return 0

# 处理点击事件（添加防抖机制）
def handle_click(clicked_node, student_data, selected_student):
    # 增强防抖机制：如果是同一个节点且时间间隔很短，则忽略
    current_time = time.time()
    if (st.session_state.last_processed_click == clicked_node and 
        current_time - st.session_state.click_timestamp < 2.0):  # 增加到2秒
        return False
    
    # 更新防抖状态
    st.session_state.last_processed_click = clicked_node
    st.session_state.click_timestamp = current_time
    
    # 判断节点层级
    node_level = get_node_level(clicked_node, student_data, selected_student)
    
    if node_level == 4:  # 第四层（题目）- 点击输出
        # 题目ID就是节点名称本身
        title_id = clicked_node
        print(f"🎯 题目被点击: {title_id}")
        st.toast(f"🎯 题目ID: {title_id}", icon="✅")
        
        # 获取题目详细信息
        title_info_data = student_data[student_data['title_ID'].astype(str) == str(title_id)]
        if not title_info_data.empty:
            title_row = title_info_data.iloc[0]
            print(f"知识点: {title_row['knowledge']}, 子知识点: {title_row['sub_knowledge']}")
            st.toast(f"知识点: {title_row['knowledge']}", icon="📋")
        
        click_record = f"{time.strftime('%H:%M:%S')} - 点击题目: {title_id}"
        st.session_state.click_history.append(click_record)
        st.session_state.clicked_data = clicked_node
        
        return True
        
    elif node_level == 3:  # 第三层（子知识点）- 点击展开/折叠
        if clicked_node in st.session_state.expanded_nodes:
            # 折叠：移除该子知识点
            st.session_state.expanded_nodes.remove(clicked_node)
            action = "折叠"
        else:
            # 展开
            st.session_state.expanded_nodes.add(clicked_node)
            action = "展开"
        
        # print(f"🌲 {action}子知识点: {clicked_node}")
        
        click_record = f"{time.strftime('%H:%M:%S')} - {action}: {clicked_node}"
        st.session_state.click_history.append(click_record)
        st.session_state.clicked_data = clicked_node
        
        return True
    
    elif node_level in [1, 2]:  # 第一、二层 - 无操作，只显示信息
        st.toast(f"ℹ️ 点击了{['', '学生', '知识点'][node_level]}: {clicked_node}", icon="ℹ️")
        return False
    
    return False

@st.cache_data
def calculate_title_stats(title_info, knowledge_mastery, all_submit_records):
    # 标准化列名
    all_submit_records.columns = all_submit_records.columns.str.strip()
    title_info.columns = title_info.columns.str.strip()
    knowledge_mastery.columns = knowledge_mastery.columns.str.strip()
    
    # 计算每个题目的提交次数
    title_submit_counts = all_submit_records.groupby('title_ID').size().reset_index(name='submit_count')
    
    # 合并题目信息
    title_stats = title_submit_counts.merge(
        title_info[['title_ID', 'knowledge', 'sub_knowledge']], 
        on='title_ID', 
        how='left'
    )
    
    # 计算每个知识点的平均掌握程度
    knowledge_mastery_avg = knowledge_mastery.groupby('knowledge_point')['mastery_level'].mean().reset_index()
    knowledge_mastery_avg.rename(columns={'knowledge_point': 'knowledge'}, inplace=True)
    
    # 合并掌握程度信息
    title_stats = title_stats.merge(
        knowledge_mastery_avg,
        on='knowledge',
        how='left'
    )
    
    # 填充缺失的掌握程度为平均值
    title_stats['mastery_level'].fillna(title_stats['mastery_level'].mean(), inplace=True)
    
    # 去除无效数据
    title_stats.dropna(subset=['knowledge', 'sub_knowledge'], inplace=True)
    
    return title_stats

# 3. 创建散点图配置
def create_scatter_chart(title_stats):
    # 计算大小映射范围
    min_mastery = title_stats['mastery_level'].min()
    max_mastery = title_stats['mastery_level'].max()
    
    # 准备散点图数据
    scatter_data = []
    for _, row in title_stats.iterrows():
        # 根据掌握程度计算点大小 (8-40)
        if max_mastery > min_mastery:
            point_size = 8 + 32 * (row['mastery_level'] - min_mastery) / (max_mastery - min_mastery)
        else:
            point_size = 20
            
        scatter_data.append(opts.ScatterItem(
            name=f"题目: {row['title_ID']}...",
            value=[row['submit_count'], row['mastery_level'], row['submit_count']],
            symbol_size=point_size,
            tooltip_opts=opts.TooltipOpts(
                formatter=f"题目ID: {row['title_ID']}<br/>"
                         f"知识点: {row['knowledge']}<br/>"
                         f"子知识点: {row['sub_knowledge']}<br/>"
                         f"提交次数: {row['submit_count']}<br/>"
                         f"掌握程度: {row['mastery_level']:.3f}"
            )
        ))
    
    # 创建散点图
    scatter = (
        Scatter(init_opts=opts.InitOpts(
            width="800px", 
            height="600px",
            bg_color="#f5f5f5"  # 浅灰色背景
        ))
        .add_xaxis([item.opts.get('value')[0] for item in scatter_data])
        .add_yaxis(
            series_name="题目分布",
            y_axis=scatter_data,
            label_opts=opts.LabelOpts(is_show=False)
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="题目信息散点图",
                title_textstyle_opts=opts.TextStyleOpts(color="#1a1a1a", font_weight="bold",font_size=25),  # 深黑色标题
                pos_left="left",
                
            ),
            xaxis_opts=opts.AxisOpts(
                name="提交次数",
                name_location="middle",
                name_gap=30,
                min_=2000,
                max_=12500,
                splitline_opts=opts.SplitLineOpts(is_show=False),
                axisline_opts=opts.AxisLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(color="#1a1a1a")),
                axistick_opts=opts.AxisTickOpts(is_show=True),
                axislabel_opts=opts.LabelOpts(color="#1a1a1a", font_weight="bold"),  # 深黑色轴标签
                name_textstyle_opts=opts.TextStyleOpts(color="#1a1a1a", font_weight="bold"),  # 轴名称颜色
                type_="value"
            ),
            yaxis_opts=opts.AxisOpts(
                name="掌握程度",
                name_location="middle",
                name_gap=50,
                min_=0.5,
                splitline_opts=opts.SplitLineOpts(is_show=False),
                axisline_opts=opts.AxisLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(color="#1a1a1a")),
                axistick_opts=opts.AxisTickOpts(is_show=True),
                axislabel_opts=opts.LabelOpts(color="#1a1a1a", font_weight="bold"),  # 深黑色轴标签
                name_textstyle_opts=opts.TextStyleOpts(color="#1a1a1a", font_weight="bold"),  # 轴名称颜色
                type_="value"
            ),
            tooltip_opts=opts.TooltipOpts(trigger="item"),
            legend_opts=opts.LegendOpts(is_show=False),
            visualmap_opts=opts.VisualMapOpts(
                type_="color",
                orient='horizontal',
                pos_left='center',
                max_=12500,
                min_=2000,
            ),
        )
    )
    
    return scatter


# 2. 创建极坐标图
def create_polar_chart2(score_data, title_id):
    # 筛选该题目的所有学生得分
    title_scores = score_data[score_data['title_ID'] == title_id]
    
    if title_scores.empty:
        return None
    
    # 按得分率分组统计人数
    score_bins = np.arange(0, 1.1, 0.1)  # 0-1分，每0.1为一个区间
    bin_labels = [f"{i:.1f}-{i+0.1:.1f}" for i in score_bins[:-1]]
    
    # 统计每个区间的人数
    title_scores['score_bin'] = pd.cut(title_scores['score_rate'], bins=score_bins, labels=bin_labels, include_lowest=True)
    bin_counts = title_scores['score_bin'].value_counts().sort_index()
    
    # 雷达图需要的指标（每个得分区间作为一个指标）
    indicators = []
    values = []
    
    for i, (bin_label, count) in enumerate(bin_counts.items()):
        indicators.append(opts.RadarIndicatorItem(name=f"得分{bin_label}", max_=500,color='#000000'))
        values.append(count)
    
    # 如果没有数据，创建默认指标
    if not indicators:
        indicators = [opts.RadarIndicatorItem(name="无数据", max_=500)]
        values = [0]
    
    # 创建雷达图
    radar = (
        Radar(init_opts=opts.InitOpts(
            width="800px", 
            height="400px",
            bg_color="#f5f5f5"  # 浅灰色背景
        ))
        .add_schema(
            schema=indicators,
            shape="circle",
            center=["50%", "60%"],  # 将Y轴位置从50%改为60%，整体向下移动
            radius="60%",
            angleaxis_opts=opts.AngleAxisOpts(
                min_=0,
                max_=360,
                is_clockwise=False,
                axistick_opts=opts.AxisTickOpts(is_show=False),
                axislabel_opts=opts.LabelOpts(is_show=True, color="#1a1a1a", font_weight="bold"),  # 深黑色标签
                splitline_opts=opts.SplitLineOpts(is_show=False),
            ),
            radiusaxis_opts=opts.RadiusAxisOpts(
                min_=0,
                max_=500,
                interval=100,
                axistick_opts=opts.AxisTickOpts(is_show=False),
                splitarea_opts=opts.SplitAreaOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(
                    is_show=True, 
                    linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.5, color="#1a1a1a")
                ),
                axislabel_opts=opts.LabelOpts(color="#1a1a1a", font_weight="bold"),  # 深黑色标签
            ),
            polar_opts=opts.PolarOpts(center=["50%", "60%"], radius=["0%", "60%"]),  # 添加极坐标系整体位置控制
            splitarea_opt=opts.SplitAreaOpts(is_show=False),
            splitline_opt=opts.SplitLineOpts(is_show=False),
        )
        .add(
            series_name=f"题目得分分布",
            data=[values],
            areastyle_opts=opts.AreaStyleOpts(opacity=0.3, color="#3498db"),  # 浅蓝色主题
            linestyle_opts=opts.LineStyleOpts(color="#2980b9", width=3),  # 深蓝色线条
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"得分率分布",
                title_textstyle_opts=opts.TextStyleOpts(color="#1a1a1a", font_weight="bold",font_size=25),  # 深黑色标题
                pos_top="5%"  # 调整标题位置，避免与下移的雷达图重叠
            ),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )
    
    return radar



# 根据掌握程度确定颜色的函数
def get_color_by_mastery(mastery_level):
    if mastery_level >= 0.9:
        return "#25A208" 
    elif mastery_level >= 0.68:
        return "#7BBF0D" 
    elif mastery_level >= 0.61:
        return "#D9BA09"   
    elif mastery_level >= 0.5:
        return "#0073FF" 
    else:
        return "#A70A0A"  

# 创建极坐标图的通用函数
def create_polar_chart(knowledge_points, mastery_levels, title):
    
    if not knowledge_points:
        return None
    
    # 先按掌握程度升序排序，使低分在内层，高分在外层
    sorted_data = sorted(zip(knowledge_points, mastery_levels), key=lambda x: x[1], reverse=False)
    sorted_knowledge_points, sorted_mastery_levels = zip(*sorted_data)
    sorted_knowledge_points = list(sorted_knowledge_points)
    sorted_mastery_levels = list(sorted_mastery_levels)
    
    # 创建数据项，每个数据项包含知识点和掌握程度
    chart_data = []
    for i in range(len(sorted_knowledge_points)):
        color = get_color_by_mastery(sorted_mastery_levels[i])
        chart_data.append({
            "value": f"{sorted_mastery_levels[i]:.2f}",
            "name": sorted_knowledge_points[i],
            "itemStyle": {"color": color}
        })
    
    # 创建极坐标图
    polar = (
        Polar(init_opts=opts.InitOpts(
            width="100px",
            height="100px",
            bg_color="#f5f5f5"  # 浅灰色背景
        ))
        .add_schema(
            radiusaxis_opts=opts.RadiusAxisOpts(
                data=sorted_knowledge_points, 
                type_="category",
                axislabel_opts=opts.LabelOpts(font_size='0'),
                
            ),
            angleaxis_opts=opts.AngleAxisOpts(
                is_clockwise=True, 
                max_=1.0,
                min_=0,
                axislabel_opts=opts.LabelOpts(color="#1a1a1a", font_weight="bold")  # 深黑色轴标签
            ),
        )
        .add(
            series_name=None,
            data=chart_data,
            type_="bar",
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=title,
                title_textstyle_opts=opts.TextStyleOpts(color="#1a1a1a", font_weight="bold")  # 深黑色标题
            )
        )
        .set_series_opts(tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{b}: {c}"))
    )
    
    return polar



# 状态分类函数
def categorize_state(state):
    if state == 'Absolutely_Correct':
        return '完全正确'
    elif state == 'Partially_Correct':
        return '部分正确'
    else:
        return '完全错误'  # 包括所有Error类型

# 颜色映射
def get_color_by_category(category):
    color_map = {
        '完全正确': '#00FF00',  # 绿色
        '部分正确': '#FFA500',  # 橙色
        '完全错误': '#FF0000'   # 红色
    }
    return color_map.get(category, '#808080')

# 创建向左柱形图函数
def create_submission_bar_chart_time(data, metric, y_axis_name):
    # 准备数据
    x_data = []
    y_data = []
    
    for i, row in data.iterrows():
        x_data.append(f"提交{i+1}")
        color = get_color_by_category(row['state_category'])
        y_data.append(opts.BarItem(
            name=f"提交{i+1}",
            value=row[metric],
            itemstyle_opts=opts.ItemStyleOpts(color=color)
        ))
    
    # 创建柱形图
    bar = (
        Bar(init_opts=opts.InitOpts(width="100px",height='100px'))
        .add_xaxis(x_data)
        .add_yaxis(y_axis_name,
                    y_data,
                    bar_width='8',
                    category_gap="10%",
                    label_opts=opts.LabelOpts(
                        is_show=True,
                        position="right",  # 对于水平柱状图，right表示柱子的右端（顶端）
                        color="black"
                    ))
        .reversal_axis()
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(name="",
                                     splitline_opts=opts.SplitLineOpts(is_show=False)  # 去除网格线
                                     ),
            yaxis_opts=opts.AxisOpts(name=y_axis_name,
                                    name_textstyle_opts=opts.TextStyleOpts(color='#000000',
                                                       font_size=20,
                                                       ),
                                    # splitline_opts=opts.SplitLineOpts(is_show=False),
                                    axislabel_opts=opts.LabelOpts(is_show=False),  # 去除网格线
                                    axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(opacity=0)), 
                                    ),
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                
            ),
            legend_opts=opts.LegendOpts(is_show=False)
        )
    )
    
    return bar

# 创建向右柱形图函数
def create_submission_bar_chart_memo(data, metric, y_axis_name):
    # 准备数据
    x_data = []
    y_data = []
    
    for i, row in data.iterrows():
        x_data.append(f"提交{i+1}")
        color = get_color_by_category(row['state_category'])
        y_data.append(opts.BarItem(
            name=f"提交{i+1}",
            value=row[metric],
            itemstyle_opts=opts.ItemStyleOpts(color=color)
        ))
    
    # 创建柱形图
    bar = (
        Bar(init_opts=opts.InitOpts(width="100px",height='200px'))
        .add_xaxis(x_data)
        .add_yaxis(
            y_axis_name,
             
            y_data,
            bar_width='8',
            category_gap="10%",
            label_opts=opts.LabelOpts(
                is_show=True,
                position="left",  # 对于水平柱状图，right表示柱子的右端（顶端）
                color="black"
            )
        )
        .reversal_axis()
        .set_global_opts(
            title_opts=opts.TitleOpts(is_show=True,title='学生题目提交信息记录',
                                      title_textstyle_opts=opts.TextStyleOpts(color="#1a1a1a", font_weight="bold",font_size=25)),  # 深黑色标题),
            
            xaxis_opts=opts.AxisOpts(
                name="",
                is_inverse=True,
                axistick_opts=opts.AxisTickOpts(is_show=False),
                
                splitline_opts=opts.SplitLineOpts(is_show=False)  # 去除网格线
            ),
            yaxis_opts=opts.AxisOpts(
                name=y_axis_name,
                axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(opacity=0)),
                axislabel_opts=opts.LabelOpts(is_show=False),
                # splitline_opts=opts.SplitLineOpts(is_show=False),  # 去除网格线
                name_textstyle_opts=opts.TextStyleOpts(color='#000000',
                                                       font_size=20,
                                                       )
            ),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(is_show=False),
            
        )
    )
    
    return bar

# 3. 提取选中学生的语言比例数据
def get_student_data(student_id,language_usage):
    student_data = language_usage[language_usage["student_ID"] == student_id].iloc[0]
    
    # 提取所有语言的ratio字段
    languages = []
    ratios = []
    for i in range(1, 6):
        lang_col = f"Language{i}_ratio"
        if lang_col in language_usage.columns:
            lang_name = f"Language{i}"
            ratio = student_data[lang_col] * 100  # 转换为百分比
            languages.append(lang_name)
            ratios.append(ratio)
    
    return languages, ratios

def create_radar_chart(languages, ratios):
    # 雷达图配置
    schema = [
        {"name": lang, "max": 50} for lang in languages
    ]
    
    radar = (
        Radar(init_opts=opts.InitOpts(bg_color="#f5f5f5"))  # 浅灰色背景
        .add_schema(
            schema=schema,
            
            splitarea_opt=opts.SplitAreaOpts(is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=0.1)),
            textstyle_opts=opts.TextStyleOpts(color="#1a1a1a", font_weight="bold"),  # 深黑色文字
            splitline_opt=opts.SplitLineOpts(
                is_show=True, 
                linestyle_opts=opts.LineStyleOpts(width=1, color="#666666")  # 深灰色分割线
            ),
            axisline_opt=opts.AxisLineOpts(
                is_show=True,
                linestyle_opts=opts.LineStyleOpts(width=2, color="#1a1a1a")  # 深黑色轴线
            )
        )
        .add(
            series_name="使用比例(%)",
            data=[ratios],
            areastyle_opts=opts.AreaStyleOpts(opacity=0.3, color="#3498db"),  # 浅蓝色主题
            linestyle_opts=opts.LineStyleOpts(width=2, color="#2980b9"),
            label_opts=opts.LabelOpts(is_show=True, formatter="{c}%", color="#1a1a1a", font_weight="bold"),
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )
    
    return radar



# 3. 雷达图生成函数（优化版）
def create_knowledge_radar(data,student_id):
    # 自定义雷达图配置
    max_mastery = 1  # 假设掌握程度是1-5级
    
    # 雷达图坐标系配置
    schema = [
        {"name": point, "max": max_mastery}
        for point in data["knowledge_points"]
    ]
    
    # 创建雷达图
    radar = (
        Radar(init_opts=opts.InitOpts(
            width="800px", 
            height="500px",
            bg_color="#f5f5f5"  # 浅灰色背景
        ))
        .add_schema(
            schema=schema,
            splitarea_opt=opts.SplitAreaOpts(
                is_show=True,
                areastyle_opts=opts.AreaStyleOpts(opacity=0.1)
            ),
            textstyle_opts=opts.TextStyleOpts(color="#1a1a1a", font_size=12, font_weight="bold"),  # 深黑色文字
            splitline_opt=opts.SplitLineOpts(
                is_show=True,
                linestyle_opts=opts.LineStyleOpts(width=1, color="#666666")  # 深灰色分割线
            ),
            axisline_opt=opts.AxisLineOpts(
                is_show=True,
                linestyle_opts=opts.LineStyleOpts(width=2, color="#1a1a1a")  # 深黑色轴线
            )
        )
        .add(
            series_name="掌握程度",
            data=[data["mastery_levels"]],
            areastyle_opts=opts.AreaStyleOpts(
                opacity=0.3,
                color="#3498db"  # 浅蓝色主题
            ),
            linestyle_opts=opts.LineStyleOpts(
                width=2,
                color="#2980b9"  # 深蓝色线条
            ),
            label_opts=opts.LabelOpts(
                is_show=False,
                formatter="{c}",
                font_size=12,
                color="#1a1a1a"
            ),
            symbol="circle",
        )
        .set_global_opts(
            legend_opts=opts.LegendOpts(is_show=False),
            tooltip_opts=opts.TooltipOpts(trigger="item")
        )
    )
    return radar


# 3. 数据筛选函数
def get_knowledge_data(student_id,main_knowledge_mastery):
    student_data = main_knowledge_mastery[main_knowledge_mastery["student_ID"] == student_id]
    if not student_data.empty:
        return {
            "knowledge_points": student_data["knowledge_point"].tolist(),
            "mastery_levels": student_data["mastery_level"].tolist()
        }
    return None

# 4. 雷达图生成函数（优化版）
def create_knowledge_radar(data,student_id):
    # 自定义雷达图配置
    max_mastery = 1  # 假设掌握程度是1-5级
    
    # 雷达图坐标系配置
    schema = [
        {"name": point, "max": max_mastery}
        for point in data["knowledge_points"]
    ]
    
    # 创建雷达图
    radar = (
        Radar(init_opts=opts.InitOpts(
            width="800px", 
            height="500px",
            bg_color="#f5f5f5"  # 浅灰色背景
        ))
        .add_schema(
            schema=schema,
            splitarea_opt=opts.SplitAreaOpts(
                is_show=True,
                areastyle_opts=opts.AreaStyleOpts(opacity=0.1)
            ),
            textstyle_opts=opts.TextStyleOpts(color="#1a1a1a", font_size=12, font_weight="bold"),  # 深黑色文字
            splitline_opt=opts.SplitLineOpts(
                is_show=True,
                linestyle_opts=opts.LineStyleOpts(width=1, color="#666666")  # 深灰色分割线
            ),
            axisline_opt=opts.AxisLineOpts(
                is_show=True,
                linestyle_opts=opts.LineStyleOpts(width=2, color="#1a1a1a")  # 深黑色轴线
            )
        )
        .add(
            series_name="掌握程度",
            data=[data["mastery_levels"]],
            areastyle_opts=opts.AreaStyleOpts(
                opacity=0.3,
                color="#3498db"  # 浅蓝色主题
            ),
            linestyle_opts=opts.LineStyleOpts(
                width=2,
                color="#2980b9"  # 深蓝色线条
            ),
            label_opts=opts.LabelOpts(
                is_show=False,
                formatter="{c}",
                font_size=12,
                color="#1a1a1a"
            ),
            symbol="circle",
        )
        .set_global_opts(
            legend_opts=opts.LegendOpts(is_show=False),
            tooltip_opts=opts.TooltipOpts(trigger="item")
        )
    )
    return radar


# 3. 数据处理
def prepare_calendar_data(student_id,student_daily_stats):
    student_data = student_daily_stats[student_daily_stats["student_ID"] == student_id]
    return [
        [str(row['date']), row['daily_study_time']]
        for _, row in student_data.iterrows()
    ]

def create_calendar_chart(data,student_id):
    if not data:
        return None
    
    # 获取日期范围
    dates = [datetime.datetime.strptime(d[0], "%Y-%m-%d").date() for d in data]
    min_date = min(dates)
    max_date = max(dates)
    max_study_time = max(d[1] for d in data)
    
    calendar = (
        Calendar(init_opts=opts.InitOpts(width="80%", height="500px",bg_color="#f5f5f5"))
        .add(
            series_name="",
            yaxis_data=data,
            
            calendar_opts=opts.CalendarOpts(
                range_=[str(min_date), str(max_date)],
                daylabel_opts=opts.CalendarDayLabelOpts(name_map="cn",label_color='#1a1a1a'),  # 深黑色日期标签
                monthlabel_opts=opts.CalendarMonthLabelOpts(name_map="cn",label_color='#1a1a1a'),  # 深黑色月份标签
                yearlabel_opts=opts.CalendarYearLabelOpts(is_show=False),
                pos_left='10%',
                pos_right='10%'
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="学生学习时长",
                title_textstyle_opts=opts.TextStyleOpts(color="#1a1a1a", font_weight="bold",font_size=25),  # 深黑色标题
                
            ),
            
            visualmap_opts=opts.VisualMapOpts(
                max_=max_study_time,
                min_=0,
                orient="horizontal",
                is_piecewise=False,

                pos_top="230px",
                pos_left="100px",
                range_color=["#e3f2fd", "#1976d2"]
                # 移除不支持的text_style参数
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="item",
                formatter="{b}: {c}分钟"
            )
        )
    )
    return calendar

# 处理数据，将学生知识点掌握情况升维到二维
def process_student_data2(df):
    if df is None:
        return None
    
    # 过滤掉21位学号的学生
    df['student_ID'] = df['student_ID'].astype(str)
    df = df[df['student_ID'].str.len() != 21]
    
    if len(df) == 0:
        st.warning("过滤后没有有效的学生数据")
        return None
    
    # 创建学生-知识点掌握矩阵
    student_knowledge_matrix = df.pivot_table(
        index='student_ID', 
        columns='knowledge_point', 
        values='mastery_level', 
        fill_value=0
    )
    
    # 计算每个学生的统计特征
    student_stats = df.groupby('student_ID').agg({
        'mastery_level': ['mean', 'std', 'count']
    }).reset_index()
    
    # 展平多级列索引
    student_stats.columns = ['student_ID', 'avg_mastery', 'mastery_std', 'knowledge_count']
    student_stats['mastery_std'] = student_stats['mastery_std'].fillna(0)
    
    # 使用PCA将知识点掌握情况降维到2维
    pca = PCA(n_components=2, random_state=42)
    knowledge_features_2d = pca.fit_transform(student_knowledge_matrix)
    
    # 调整PCA结果，让掌握程度高的学生显示在右侧
    # 计算第一主成分与平均掌握水平的相关性，如果为负则翻转
    correlation = np.corrcoef(knowledge_features_2d[:, 0], student_stats['avg_mastery'])[0, 1]
    if correlation < 0:
        knowledge_features_2d[:, 0] = -knowledge_features_2d[:, 0]
    
    student_stats['pca_dim1'] = knowledge_features_2d[:, 0]
    student_stats['pca_dim2'] = knowledge_features_2d[:, 1]
    
    return student_stats, pca.explained_variance_ratio_



#! 以下是页面设置
# 页面标题 - 使用更小的标题
# st.markdown("### 数据可视化大屏")
# 在容器底部添加填充元素（可选）

# 添加浅灰色主题CSS样式
st.markdown("""
<style>
/* 全局浅灰色主题 */
.stApp {
    background-color: #ffffff;
    color: #ffffff;
}

/* 炫酷标题样式 */
.cool-title {
    text-align: center;
    padding: 8px 0;
    margin: 5px 0 15px 0;
    width: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #667eea 100%);
    border-radius: 8px;
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(10px);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.cool-title::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    animation: shine 4s ease-in-out infinite;
}

.cool-title::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(circle at 50% 50%, rgba(255,255,255,0.1) 0%, transparent 70%);
    animation: pulse-glow 3s ease-in-out infinite alternate;
}

@keyframes shine {
    0% { left: -100%; }
    50% { left: -100%; }
    100% { left: 100%; }
}

@keyframes pulse-glow {
    0% { opacity: 0.3; transform: scale(0.95); }
    100% { opacity: 0.7; transform: scale(1.05); }
}

.cool-title h1 {
    font-size: 2rem !important;
    font-weight: 600 !important;
    color: #ffffff !important;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3) !important;
    margin: 0 !important;
    font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif !important;
    letter-spacing: 3px !important;
    position: relative;
    z-index: 2;
    text-align: center;
}

.cool-title .subtitle {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.85);
    margin-top: 2px;
    font-weight: 300;
    letter-spacing: 1px;
    position: relative;
    z-index: 2;
    text-align: center;
}

.cool-title .decorative-line {
    width: 600px;
    height: 3px;
    background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #ff6b6b);
    margin: 4px auto;
    border-radius: 1px;
    animation: rainbow-flow 2s linear infinite;
    position: relative;
    z-index: 2;
    align-self: center;
}

@keyframes rainbow-flow {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

.cool-title .decorative-line {
    background-size: 200% 100%;
}

/* 直接修饰Streamlit的列容器 */
.stColumn {
    position: relative;
}

.stColumn > div {
    border: 3px solid transparent;
    border-radius: 12px;
    background: linear-gradient(#f5f5f5, #f5f5f5) padding-box, 
                linear-gradient(135deg, #667eea, #764ba2, #3498db, #667eea) border-box;
    background-size: 400% 400%;
    animation: border-animation 4s ease-in-out infinite;
    padding: 15px;
    margin: 5px;
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
}

/* 为中间列添加特殊样式 */
.stColumn:nth-child(2) > div {
    border-width: 4px;
    box-shadow: 0 6px 25px rgba(102, 126, 234, 0.2);
}

/* 取消桑基图和圆环图容器的边框 - 针对kg_col1和kg_col2 */
.stColumn:nth-child(2) > div > div:first-child .stColumn > div {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    animation: none !important;
    padding: 5px !important;
    margin: 0 !important;
}

.stColumn:nth-child(2) > div > div:first-child .stColumn > div::before {
    display: none !important;
}

/* 保持第二行柱状图的边框 */
.stColumn:nth-child(2) > div > div:nth-child(2) {
    border: 3px solid transparent;
    border-radius: 12px;
    background: linear-gradient(#f5f5f5, #f5f5f5) padding-box, 
                linear-gradient(135deg, #667eea, #764ba2, #3498db, #667eea) border-box;
    background-size: 400% 400%;
    animation: border-animation 4s ease-in-out infinite;
    padding: 15px;
    margin: 5px;
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
}

/* 动态边框动画 */
@keyframes border-animation {
    0% { 
        background-position: 0% 50%;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
    }
    25% { 
        background-position: 100% 50%;
        box-shadow: 0 6px 25px rgba(118, 75, 162, 0.2);
    }
    50% { 
        background-position: 100% 100%;
        box-shadow: 0 8px 30px rgba(52, 152, 219, 0.25);
    }
    75% { 
        background-position: 0% 100%;
        box-shadow: 0 6px 25px rgba(118, 75, 162, 0.2);
    }
    100% { 
        background-position: 0% 50%;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
    }
}

/* 特殊的发光效果 */
.stColumn > div::before {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    border-radius: 14px;
    background: linear-gradient(135deg, #667eea, #764ba2, #3498db, #667eea);
    background-size: 400% 400%;
    animation: glow-border 3s linear infinite;
    z-index: -1;
    filter: blur(6px);
    opacity: 0.7;
}

/* 取消桑基图和圆环图列的发光效果 */
.stColumn:nth-child(2) > div > div:first-child .stColumn > div::before {
    display: none !important;
}

@keyframes glow-border {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

/* 标题样式 */
h1, h2, h3, h4, h5, h6 {
    color: #2980b9 !important;
    text-shadow: 0 0 5px rgba(41, 128, 185, 0.2);
}

/* 选择框浅灰色样式 */
.stSelectbox > div > div {
    background-color: #ffffff;
    color: #2c3e50;
    border: 1px solid #2980b9;
    border-radius: 8px;
}

/* Tab标签浅灰色样式 - 修改为背景色 */
.stTabs [data-baseweb="tab-list"] {
    background-color: #f5f5f5 !important;
    border-radius: 8px;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    color: #2c3e50 !important;
    border-radius: 6px;
    margin: 2px;
}

/* Tab内容区域背景色 */
.stTabs [data-baseweb="tab-panel"] {
    background-color: #f5f5f5 !important;
    border-radius: 0 0 8px 8px;
    padding: 10px;
}

/* 指标卡片浅灰色样式 */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    border: 1px solid #2980b9;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 4px 15px rgba(41, 128, 185, 0.1);
}

div[data-testid="metric-container"] > div {
    color: #2c3e50 !important;
}

/* 警告和信息框浅灰色样式 */
.stAlert {
    background-color: #ffffff;
    color: #2c3e50;
    border-left: 4px solid #2980b9;
    border-radius: 8px;
}

/* 复选框样式 */
.stCheckbox > div {
    color: #2c3e50;
}

/* 滑块样式 */
.stSlider > div {
    color: #2c3e50;
}

/* 侧边栏样式 */
.css-1d391kg {
    background-color: #ffffff;
}

# </style>
""", unsafe_allow_html=True)

# 炫酷标题HTML
st.markdown("""
<div class="cool-title">
    <h1>析数启智</h1>
    <div class="decorative-line"></div>
    <div class="subtitle">时序多变量教育数据可视分析平台</div>
</div>
""", unsafe_allow_html=True)



#! 以下是页面设计
# 创建一个容器来限制整体内容区域
with st.container():
    # 创建三列布局，中间列宽度是两侧的2倍
    col1, col2, col3 = st.columns([1, 2, 1])
    student_info, title_info, knowledge_mastery, all_submit_records,submission_details,language_usage,main_knowledge_mastery,score_rate_data,student_daily_stats = load_data()

    # 第一列：3个图表
    
    # 第二列：2个大图表
    with col2:
        # 第一行 - 桑基图和圆环图
        with st.container():
            # 创建两列布局用于放置知识图谱和弧形仪表盘
            kg_col1, kg_col2 = st.columns([2, 1])
            
            correct_records = preprocess_data(student_info, title_info, knowledge_mastery, all_submit_records)
            
            with kg_col2:#放置两个圆环图表示学生的掌握
                # 获取当前学生ID，优先从session state获取
                current_selected_student = st.session_state.get('student_id', sorted(correct_records['student_ID'].unique())[0])
                
                # 创建选择框，使用当前选中的学生作为默认值
                available_students = sorted(correct_records['student_ID'].unique())
                try:
                    default_index = available_students.index(current_selected_student)
                except ValueError:
                    default_index = 0
                    current_selected_student = available_students[0]
                
                student_id = st.selectbox(
                    "🎯 选择学生ID", 
                    available_students,
                    index=default_index,
                    label_visibility="visible",
                    key="student_selectbox"
                )
                
                # 只有当选择的学生ID与session state中的不同时才更新
                if student_id != st.session_state.get('student_id'):
                    st.session_state.student_id = student_id
                
                student_data1 = knowledge_mastery[knowledge_mastery['student_ID'] == student_id]
                knowledge_points = student_data1['knowledge_point'].tolist()
                mastery_levels = student_data1['mastery_level'].tolist()
                # # 添加排序选项
                # sort_by_mastery = st.checkbox("按掌握程度排序（高分在外层）", value=False)

                # # 如果选择排序，则重新排列数据
                # if sort_by_mastery:
                #     # 创建数据对并按掌握程度降序排序
                #     sorted_data = sorted(zip(odd_knowledge_points, odd_mastery_levels), key=lambda x: x[1], reverse=True)
                #     odd_knowledge_points, odd_mastery_levels = zip(*sorted_data)
                #     odd_knowledge_points = list(odd_knowledge_points)
                #     odd_mastery_levels = list(odd_mastery_levels)

                # 分离奇数和偶数索引的数据
                odd_indices = [i for i in range(len(knowledge_points)) if i % 2 == 1]  # 奇数索引
                even_indices = [i for i in range(len(knowledge_points)) if i % 2 == 0]  # 偶数索引

                odd_knowledge_points = [knowledge_points[i] for i in odd_indices]
                odd_mastery_levels = [mastery_levels[i] for i in odd_indices]

                even_knowledge_points = [knowledge_points[i] for i in even_indices]
                even_mastery_levels = [mastery_levels[i] for i in even_indices]

                # 对奇数和偶数数据分别按掌握程度排序（高分在外侧）
                if odd_knowledge_points:
                    sorted_odd = sorted(zip(odd_knowledge_points, odd_mastery_levels), key=lambda x: x[1], reverse=True)
                    odd_knowledge_points, odd_mastery_levels = zip(*sorted_odd)
                    odd_knowledge_points = list(odd_knowledge_points)
                    odd_mastery_levels = list(odd_mastery_levels)

                if even_knowledge_points:
                    sorted_even = sorted(zip(even_knowledge_points, even_mastery_levels), key=lambda x: x[1], reverse=True)
                    even_knowledge_points, even_mastery_levels = zip(*sorted_even)
                    even_knowledge_points = list(even_knowledge_points)
                    even_mastery_levels = list(even_mastery_levels)
                # 创建奇数索引图表
                if odd_knowledge_points:
                    polar_odd = create_polar_chart(odd_knowledge_points, odd_mastery_levels, "")
                    if polar_odd:
                        st_pyecharts(polar_odd, height='250px',width='320px')
                # 创建偶数索引图表
                if even_knowledge_points:
                    polar_even = create_polar_chart(even_knowledge_points, even_mastery_levels, "")
                    if polar_even:
                        st_pyecharts(polar_even, height='250px',width='320px')

        with kg_col1:
            processed_data = preprocess_data2(student_info, title_info, knowledge_mastery, all_submit_records)
            student_data = processed_data[processed_data['student_ID'] == student_id]
            if not student_data.empty:
                # 创建桑基图配置
                options = create_sankey_options(student_data, student_id)
                
                # 使用st_echarts显示图表并捕获点击事件
                clicked_sankey_node = st_echarts(
                    options=options,
                    events={
                        "click": "function(params) { return params.name; }"
                    },
                    height="600px",
                    key="sankey_chart"
                )

                # 处理桑基图点击事件 - 添加更严格的条件
                if (clicked_sankey_node and 
                    clicked_sankey_node != st.session_state.get('last_sankey_click', None)):
                    
                    # 设置最后点击的节点，避免重复处理
                    st.session_state.last_sankey_click = clicked_sankey_node
                    
                    if handle_click(clicked_sankey_node, student_data, student_id):
                        # 如果是题目节点，更新选中的问题ID
                        node_level = get_node_level(clicked_sankey_node, student_data, student_id)
                        if node_level == 4:  # 题目节点
                            st.session_state.selected_question_id = clicked_sankey_node
                        st.rerun()

            else:
                st.warning("该学生没有答题记录")

        # 第二行 - 柱状图部分
        with st.container():
            submission_details['state_category'] = submission_details['submission_state'].apply(categorize_state)
            
            # 根据选择的学生筛选题目
            student_submission_data = submission_details[submission_details['student_ID'] == student_id]
            
            # 确定要显示的问题ID（优先使用桑基图中选中的题目，否则使用默认值）
            display_question_id = st.session_state.selected_question_id
            if display_question_id is None:
                # 使用该学生的第一个问题作为默认值
                available_questions = student_submission_data['question_ID'].unique()
                if len(available_questions) > 0:
                    display_question_id = available_questions[0]
                    st.session_state.selected_question_id = display_question_id
            
            # 筛选数据
            if display_question_id:
                filtered_data = student_submission_data[
                    student_submission_data['question_ID'] == display_question_id
                ].reset_index(drop=True)
                
                if not filtered_data.empty:
                    # 创建两个柱状图
                    memory_chart = create_submission_bar_chart_memo(
                        filtered_data, 
                        'memory_usage',  
                        "内存使用 (KB)"
                    )

                    time_chart = create_submission_bar_chart_time(
                        filtered_data, 
                        'time_consumption', 
                        "时间消耗 (秒)"
                    )
                    
                    grid = Grid()
                    grid.add(time_chart, grid_opts=opts.GridOpts(pos_left="51%",pos_top='20%'))
                    grid.add(memory_chart, grid_opts=opts.GridOpts(pos_right="51%",pos_top='20%'))
                    st_pyecharts(grid, height=360)
                else:
                    st.info(f"学生 {student_id} 在问题 {display_question_id} 上没有提交记录")
            else:
                st.info(f"学生 {student_id} 没有任何提交记录")

    
    with col3:
        with st.container():
            # 预处理数据以获取学生ID
            correct_records = preprocess_data(student_info, title_info, knowledge_mastery, all_submit_records)
            
            # 获取学生ID（从第二列的选择器获取，如果没有则使用第一个）
            if 'student_id' not in st.session_state:
                st.session_state.student_id = correct_records['student_ID'].iloc[0]
            
            
            
            # 计算统计信息
            title_stats = calculate_title_stats(title_info, knowledge_mastery, all_submit_records)
            
            # 创建散点图
            scatter = create_scatter_chart(title_stats)
            
            # 显示散点图并捕获点击事件
            clicked_title = st_pyecharts(
                scatter, 
                height=300,
                events={
                    "click": "function(params) { return params.name.replace('题目: ', '').replace('...', ''); }"
                },
                key="scatter_chart"
            )
            
            # 处理散点图点击事件 - 添加更严格的防抖
            if (clicked_title and 
                clicked_title != st.session_state.selected_title_id and
                clicked_title != st.session_state.get('last_scatter_click', None)):
                st.session_state.selected_title_id = clicked_title
                st.session_state.last_scatter_click = clicked_title
                st.rerun()
            
            # 使用会话状态中的选中题目ID
            current_title_id = st.session_state.selected_title_id
            
            score_rate_data.columns = score_rate_data.columns.str.strip()
            radar = create_polar_chart2(score_rate_data, current_title_id)
            st_pyecharts(radar, height=400)
            # 加载数据
            raw_data = knowledge_mastery

            if raw_data is not None:
                # 处理数据
                student_stats, pca_variance_ratio = process_student_data2(raw_data)
                
                
                # Plotly散点图 - 使用平均掌握水平作为颜色
                fig_plotly = px.scatter(
                    student_stats,
                    x='pca_dim1',
                    y='pca_dim2',
                    color='avg_mastery',
                    hover_data=['student_ID', 'avg_mastery', 'mastery_std', 'knowledge_count'],
                    labels={
                        'pca_dim1': 'PCA维度1 (掌握程度 →)',
                        'pca_dim2': 'PCA维度2',
                        'avg_mastery': '平均掌握水平'
                    },
                    title="学生掌握情况分布图",
                    
                    color_continuous_scale='viridis_r'  # 反转颜色，深色表示掌握好
                )
                
                # 更新散点样式：禁用选中时其他点变暗
                fig_plotly.update_traces(
                    unselected=dict(marker=dict(opacity=1))  # 保持未选中点的透明度为1

                )
                
                fig_plotly.update_layout(
                    height=300,
                    title_font=dict(size=25),
                    clickmode='event+select',  # 启用点击事件
                    plot_bgcolor='#f5f5f5',  # 图表背景为灰色
                    paper_bgcolor='#f5f5f5',  # 整个图形背景为灰色
                    xaxis=dict(
                        showgrid=False,  # 去掉X轴网格线
                        zeroline=False   # 去掉零线
                    ),
                    yaxis=dict(
                        showgrid=False,  # 去掉Y轴网格线
                        zeroline=False   # 去掉零线
                    ),
                    coloraxis_showscale=False  # 去掉颜色条
                )
                
                # 显示图表并捕获点击事件
                clicked_data = st.plotly_chart(fig_plotly, use_container_width=True, on_select="rerun")
                
                # 处理点击事件，返回学生ID
                if clicked_data and 'selection' in clicked_data and 'points' in clicked_data['selection']:
                    selected_points = clicked_data['selection']['points']
                    if selected_points:
                        # 获取点击的点的索引
                        point_indices = [point['point_index'] for point in selected_points]
                        selected_students = student_stats.iloc[point_indices]['student_ID'].tolist()
                        
                        #! 更新session state中的学生ID
                        if selected_students:
                            new_student_id = selected_students[0]
                            # 只有当点击的学生ID与当前不同时才更新和重新运行
                            if new_student_id != st.session_state.get('student_id'):
                                st.session_state.student_id = new_student_id
                                st.session_state.selected_student_ids = selected_students
                                # print(f"通过散点图选择学生: {new_student_id}")
                                st.rerun()
                        
                        # 在控制台输出（用于调试）
                        # print(selected_students[0])

        st.markdown('</div>', unsafe_allow_html=True)

    
    # 第三列：3个图表
    with col1:
        
        # # 日历图也需要修复暗色主题
        # # 日历图也使用固定数据
        # c = (
        #     Calendar(init_opts=opts.InitOpts(bg_color="#1e1e2e"))  # 添加暗色背景
        #     .add(
        #         "",
        #         st.session_state.calendar_data,
        #         calendar_opts=opts.CalendarOpts(
        #             range_=["2024-09-01", "2025-01-01"],
        #             daylabel_opts=opts.CalendarDayLabelOpts(name_map="cn"),
        #             monthlabel_opts=opts.CalendarMonthLabelOpts(name_map="cn"),
        #             yearlabel_opts=opts.CalendarYearLabelOpts(is_show=False),
        #             pos_left='10px',

        #         ),
        #     )
        #     .set_global_opts(
        #         title_opts=opts.TitleOpts(
        #             title="2024年9月-2025年1月微信步数情况",
        #             title_textstyle_opts=opts.TextStyleOpts(color="#ffffff")
        #         ),
        #         visualmap_opts=opts.VisualMapOpts(
        #             max_=20000,
        #             min_=500,
        #             orient="horizontal",
        #             is_piecewise=True,
        #             pos_top="230px",
        #             pos_left="100px",
        #         ),
                
        #     )
        # )
        # st_pyecharts(c,width='400px')
        # 显示当前选中学生的基本信息
        current_student_id = st.session_state.get('student_id', correct_records['student_ID'].iloc[0])
        student_basic_info = student_info[student_info['student_ID'] == current_student_id]
        
        if not student_basic_info.empty:
            info_row = student_basic_info.iloc[0]
            
            # 使用自定义HTML显示学生信息
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                border: 1px solid #2980b9;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 4px 15px rgba(41, 128, 185, 0.1);
            ">
                <h4 style="color: #2980b9; margin-bottom: 15px; text-align: center;">👤 学生信息</h4>
                <div style="color: #2c3e50; line-height: 1.8;">
                    <div><strong>学号:</strong> """ + str(info_row['student_ID']) + """</div>
                    <div><strong>性别:</strong> """ + str(info_row['sex']) + """</div>
                    <div><strong>年龄:</strong> """ + str(info_row['age']) + """岁</div>
                    <div><strong>专业:</strong> """ + str(info_row['major']) + """</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with st.container():
            # 使用session state中的student_id
            current_student_id = st.session_state.get('student_id', student_id)
            calendar_data = prepare_calendar_data(current_student_id, student_daily_stats)
            
            # 5. 显示图表
            if calendar_data:
                calendar_chart = create_calendar_chart(calendar_data,student_id)
                if calendar_chart:
                    selected_date = st_pyecharts(
                        calendar_chart,
                        events={"click": "function(params){return params.value[0]}"},
                        height="300px",
                        width='550px',
                    )
                    
                else:
                    st.warning("无法生成日历图")
        

        st.markdown('</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["编程语言", "学习特征", "知识掌握"])
        with tab1:
            # 使用session state中的student_id，确保与散点图点击同步
            current_student_id = st.session_state.get('student_id', student_id)
            languages, ratios = get_student_data(current_student_id, language_usage)

            radar_chart = create_radar_chart(languages, ratios)
            st_pyecharts(radar_chart, height="350px",width='500px')
        
        with tab2:
            # 学生每日学习情况雷达图 - 从fig7.py集成
            def get_student_stats(df, student_id, date):
                # 添加调试信息
                print(f"查找学生ID: {student_id}, 类型: {type(student_id)}")
                print(f"查找日期: {date}, 类型: {type(date)}")
                
                # 确保student_ID类型匹配
                df['student_ID'] = df['student_ID'].astype(str)
                student_id = str(student_id)
                
                # 分步筛选以便调试
                student_records = df[df['student_ID'] == student_id]
                print(f"找到该学生的记录数: {len(student_records)}")
                
                if len(student_records) > 0:
                    print(f"该学生的可用日期: {student_records['date'].unique()}")
                    
                # 如果传入的日期是None或者筛选后为空，使用最新日期
                if date is None or len(student_records) == 0:
                    if len(student_records) > 0:
                        date = student_records['date'].max()
                        print(f"使用最新日期: {date}")
                    else:
                        return None
                
                # 最终筛选
                stats = student_records[student_records['date'] == date]
                print(f"最终匹配的记录数: {len(stats)}")
                
                if not stats.empty:
                    result = {
                        "daily_study_time": stats['daily_study_time'].values[0],
                        "daily_submissions": stats['daily_submissions'].values[0],
                        "avg_score": stats['avg_score'].values[0]
                    }
                    print(f"返回数据: {result}")
                    return result
                return None
            
            def create_radar_chart3(data, student_id, selected_date):
                # 自定义每个指标的最大值
                max_values = {
                    "study_time": 600,  # 学习时间最大600分钟（10小时）
                    "submissions": 30,  # 提交次数最大30次
                    "score": 100        # 分数最大100分
                }
                
                # 雷达图坐标系配置
                schema = [
                    {"name": "每日学习时间(分钟)", "max": 500},
                    {"name": "每日提交次数", "max": 60},
                    {"name": "平均得分", "max": 1}
                ]
                
                # 实际数据点 - 确保数据类型正确
                study_time = float(data['daily_study_time']) if data['daily_study_time'] is not None else 0
                submissions = float(data['daily_submissions']) if data['daily_submissions'] is not None else 0
                avg_score = float(data['avg_score']) if data['avg_score'] is not None else 0
                
                radar_data = [
                    [
                        min(study_time, max_values["study_time"]),
                        min(submissions, max_values["submissions"]),
                        min(avg_score, max_values["score"])
                    ]
                ]
                
                print(f"雷达图数据: {radar_data}")  # 调试输出
                
                # 创建雷达图
                radar = (
                    Radar(init_opts=opts.InitOpts(width="800px", height="500px", bg_color="#f5f5f5"))
                    .add_schema(
                        schema=schema,
                        splitarea_opt=opts.SplitAreaOpts(
                            is_show=True,
                            areastyle_opts=opts.AreaStyleOpts(opacity=0.3)
                        ),
                        textstyle_opts=opts.TextStyleOpts(color="#1a1a1a", font_size=12, font_weight="bold"),  # 深黑色文字
                        splitline_opt=opts.SplitLineOpts(
                            is_show=True, 
                            linestyle_opts=opts.LineStyleOpts(width=1, color="#666666")  # 深灰色分割线
                        ),
                        axisline_opt=opts.AxisLineOpts(
                            is_show=True,
                            linestyle_opts=opts.LineStyleOpts(width=2, color="#1a1a1a")  # 深黑色轴线
                        )
                    )
                    .add(
                        series_name="指标值",
                        data=radar_data,
                        areastyle_opts=opts.AreaStyleOpts(
                            opacity=0.5,
                            color="#5470C6"
                        ),
                        linestyle_opts=opts.LineStyleOpts(
                            width=3,
                            color="#5470C6"
                        ),
                        label_opts=opts.LabelOpts(
                            is_show=True,
                            formatter="{c}",
                            font_size=14,
                            color="#1a1a1a",
                            font_weight="bold"
                        ),
                    )
                    .set_global_opts(
                        legend_opts=opts.LegendOpts(is_show=False),
                    )
                )
                return radar
            
            # 获取当前选择的日期和学生ID
            current_student_id = st.session_state.get('student_id', student_id)
            # 如果没有选择日期（从日历图点击），使用最近的可用日期
            if 'selected_date' not in locals() or selected_date is None:
                # 先获取该学生的数据，再取最新日期
                student_records = student_daily_stats[student_daily_stats['student_ID'].astype(str) == str(current_student_id)]
                if not student_records.empty:
                    selected_date = student_records['date'].max()
                else:
                    selected_date = None
            
            # 添加调试开关，默认关闭
            show_debug = False
            if show_debug:
                st.write(f"学号: {current_student_id}, 类型: {type(current_student_id)}")
                st.write(f"日期: {selected_date}, 类型: {type(selected_date)}")

                # 检查数据中是否存在该学生该日期的记录
                matching_records = student_daily_stats[
                    (student_daily_stats['student_ID'] == current_student_id)
                ]
                st.write(f"该学生共有 {len(matching_records)} 条记录")
                if not matching_records.empty:
                    st.write("可用日期:")
                    st.write(matching_records['date'].unique())
            
            # 确保日期类型匹配 - 不显示调试信息
            if selected_date is not None and not isinstance(selected_date, type(student_daily_stats['date'].iloc[0])):
                try:
                    # 如果是字符串，尝试转换
                    if isinstance(selected_date, str):
                        import datetime
                        selected_date = datetime.datetime.strptime(selected_date, "%Y-%m-%d").date()
                    # 如果是pandas Timestamp，转换为date
                    elif hasattr(selected_date, 'date'):
                        selected_date = selected_date.date()
                except Exception as e:
                    if show_debug:
                        st.error(f"日期转换错误: {e}")
            
            # 获取学生统计数据
            stats = get_student_stats(student_daily_stats, current_student_id, selected_date)
            
            # 显示结果
            if stats:
                # 显示雷达图
                radar_chart = create_radar_chart3(stats, current_student_id, selected_date)
                st_pyecharts(radar_chart, height='350px', width='500px')
                
                # # 显示详细数据
                # st.markdown(f"""
                # <div style="font-size: 14px; color: #2c3e50; margin-top: 10px;">
                # - 学习时间: {stats['daily_study_time']} 分钟
                # - 提交次数: {stats['daily_submissions']} 次
                # - 平均得分: {stats['avg_score']:.1f} 分
                # </div>
                # """, unsafe_allow_html=True)
            else:
                st.warning(f"未找到学号 {current_student_id} 在 {selected_date} 的记录")
                # 添加进一步的错误诊断
                matching_records = student_daily_stats[student_daily_stats['student_ID'].astype(str) == str(current_student_id)]
                if len(matching_records) > 0:
                    st.info("请检查日期格式是否匹配。尝试选择上面列出的可用日期之一。")

        with tab3:
            # 使用 session state 中的 student_id，确保与散点图点击同步
            current_student_id = st.session_state.get('student_id', student_id)
            knowledge_data = get_knowledge_data(current_student_id, main_knowledge_mastery)
            if knowledge_data:
                radar_chart = create_knowledge_radar(knowledge_data, current_student_id)
                st_pyecharts(radar_chart, height="351px",width='500px')

