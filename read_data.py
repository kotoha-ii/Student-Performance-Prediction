import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_oulad(data_dir):
    """
    加载OULAD数据集中的所有CSV文件到Pandas DataFrame
    
    参数:
        data_dir (str): 包含OULAD数据文件的目录路径
        
    返回:
        dict: 包含以下键的字典:
            'student_info': 学生基本信息
            'courses': 课程信息
            'student_registration': 学生注册信息
            'assessments': 评估任务信息
            'student_assessment': 学生评估结果
            'student_vle': 学生VLE活动记录
            'vle': VLE资源信息
    
    数据结构:
        每个键对应一个DataFrame,包含原始CSV文件的数据
    """
    # 定义所有需要的文件名
    files = {
        'student_info': 'studentInfo.csv',
        'courses': 'courses.csv',
        'student_registration': 'studentRegistration.csv',
        'assessments': 'assessments.csv',
        'student_assessment': 'studentAssessment.csv',
        'student_vle': 'studentVle.csv',
        'vle': 'vle.csv'
    }
    dfs = {}
    for key, fname in files.items():
        # 构建完整文件路径
        path = os.path.join(data_dir, fname)
        # 检查文件是否存在
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Expected file not found: {path}")
        # 读取CSV文件
        dfs[key] = pd.read_csv(path)
    return dfs

def compute_week(df_vle):
    """
    将日期列转换为周数
    
    参数:
        df_vle (DataFrame): 包含学生VLE活动记录的DataFrame
        
    返回:
        DataFrame: 添加了'week'列的相同DataFrame
        
    处理说明:
        'date'列表示自课程开始以来的天数
        'week'计算为 date // 7,表示周数索引(0开始)
    """
    if 'date' in df_vle.columns:
        # 复制DataFrame避免修改原始数据
        df_vle = df_vle.copy()
        # 计算周数
        df_vle['week'] = (df_vle['date'] // 7).astype(int)
    else:
        raise ValueError("Column 'date' not found in studentVle; cannot compute week.")
    return df_vle

def aggregate_behavior_features(dfs, n_weeks=4):
    """
    聚合前n_weeks的学生行为特征
    
    参数:
        dfs (dict): 包含所有数据集的字典
        n_weeks (int): 考虑的行为周数(默认前4周)
        
    返回:
        DataFrame: 包含以下特征的DataFrame,索引为(id_student, code_module, code_presentation):
            - total_clicks: 总点击次数
            - active_weeks: 有活动的周数
            - login_days: 有登录的天数
            - clicks_<activity_type>: 各种活动类型的点击次数
            - n_expected_assess: 课程预期作业数
            - n_attempts: 学生实际尝试作业数
            - n_missing: 未完成的作业数
            - avg_score: 平均分数
    
    处理说明:
        1. 计算前n_weeks的每周活动数据
        2. 统计学生行为指标(点击、登录、活跃度)
        3. 按活动类型分类统计点击次数
        4. 合并评估数据(作业完成情况、平均分)
    """
    # 获取VLE活动数据
    df_vle = dfs['student_vle']
    
    # 计算周数
    df_vle = compute_week(df_vle)
    
    # 过滤前n_weeks的数据
    df_vle_early = df_vle[df_vle['week'] < n_weeks]
    
    # 1. 计算基本行为指标
    
    # 总点击次数
    feat_clicks = df_vle_early.groupby(
        ['id_student', 'code_module', 'code_presentation']
    )['sum_click'].sum().rename('total_clicks')
    
    # 活跃周数(有活动的周数)
    weeks_active = df_vle_early[df_vle_early['sum_click'] > 0].groupby(
        ['id_student', 'code_module', 'code_presentation']
    )['week'].nunique().rename('active_weeks')
    
    # 登录天数(近似统计)
    df_vle_early = df_vle_early.copy()
    df_vle_early['login_day'] = df_vle_early['date']  # 直接使用date作为登录日标识
    login_days = df_vle_early.groupby(
        ['id_student', 'code_module', 'code_presentation']
    )['login_day'].nunique().rename('login_days')
    
    # 2. 按活动类型分类统计
    
    # 获取活动类型元数据
    df_vle_meta = dfs['vle'][['id_site', 'activity_type']].drop_duplicates()
    # 合并活动类型信息
    df_vle_early = df_vle_early.merge(df_vle_meta, on='id_site', how='left')
    
    # 创建活动类型点击次数透视表
    type_counts = df_vle_early.pivot_table(
        index=['id_student', 'code_module', 'code_presentation'],
        columns='activity_type',
        values='sum_click',
        aggfunc='sum',
        fill_value=0
    )
    # 重命名列(格式：clicks_<activity_type>)
    type_counts.columns = [f"clicks_{str(col)}" for col in type_counts.columns]
    
    # 合并行为特征
    df_feat = pd.concat([feat_clicks, weeks_active, login_days, type_counts], axis=1).reset_index()
    df_feat = df_feat.set_index(['id_student', 'code_module', 'code_presentation'])
    
    # 3. 处理评估数据
    
    # 获取评估元数据
    df_assess_meta = dfs['assessments'][['id_assessment', 'code_module', 'code_presentation']]
    
    # 计算每门课的预期作业数量
    expected_counts = df_assess_meta.groupby(
        ['code_module', 'code_presentation']
    )['id_assessment'].nunique().reset_index(name='n_expected_assess')
    
    # 创建基础数据框架(包含所有学生-课程组合)
    base_df = df_feat.reset_index()[['id_student', 'code_module', 'code_presentation']]
    
    # 合并预期作业数量
    base_df = base_df.merge(
        expected_counts, 
        on=['code_module', 'code_presentation'], 
        how='left'
    )
    
    # 计算学生实际完成的作业数量
    # 合并学生评估和课程信息
    student_assessments = dfs['student_assessment'].merge(
        df_assess_meta, 
        on='id_assessment', 
        how='left'
    )
    # 统计每个学生完成的作业数量
    actual_counts = student_assessments.groupby(
        ['id_student', 'code_module', 'code_presentation']
    )['id_assessment'].nunique().reset_index(name='n_attempts')
    
    # 合并实际完成数量到基础框架
    base_df = base_df.merge(
        actual_counts, 
        on=['id_student', 'code_module', 'code_presentation'], 
        how='left'
    )
    
    # 计算缺失的作业数量
    base_df['n_attempts'] = base_df['n_attempts'].fillna(0)
    base_df['n_missing'] = base_df['n_expected_assess'] - base_df['n_attempts']
    
    # 计算学生平均成绩
    avg_score = student_assessments.groupby(
        ['id_student', 'code_module', 'code_presentation']
    )['score'].mean().reset_index(name='avg_score')
    
    # 合并平均成绩到基础框架
    base_df = base_df.merge(
        avg_score, 
        on=['id_student', 'code_module', 'code_presentation'], 
        how='left'
    )
    
    # 设置多级索引
    base_df = base_df.set_index(['id_student', 'code_module', 'code_presentation'])
    
    # 4. 将评估数据合并到行为特征框架
    df_feat = df_feat.join(base_df[['n_expected_assess', 'n_attempts', 'n_missing', 'avg_score']])
    
    # 5. 填充缺失值
    # 对于没有评估记录的学生,设置默认值
    df_feat['avg_score'] = df_feat['avg_score'].fillna(0)  # 成绩设为0
    df_feat['n_expected_assess'] = df_feat['n_expected_assess'].fillna(0)  # 预期作业设为0
    df_feat['n_attempts'] = df_feat['n_attempts'].fillna(0)  # 实际完成设为0
    df_feat['n_missing'] = df_feat['n_missing'].fillna(0)  # 缺失作业设为0
    
    return df_feat.reset_index()

def generate_labels_simple(df_info):
    """
    生成简单困难标签(通过/不及格)
    
    参数:
        df_info (DataFrame): 学生信息数据
        
    返回:
        DataFrame: 包含以下列的DataFrame:
            - id_student, code_module, code_presentation: 学生和课程标识
            - label_simple: 二元标签(0=通过/优等,1=不及格/退学)
    """
    df = df_info.copy()
    # 定义标签转换函数
    def label_fn(x):
        if x in ['Fail', 'Withdrawn']: return 1  # 困难学生
        elif x in ['Pass', 'Distinction']: return 0  # 非困难学生
        else: return np.nan  # 无效结果
    
    # 应用标签转换
    df['label_simple'] = df['final_result'].apply(label_fn)
    # 移除无效结果
    df = df.dropna(subset=['label_simple'])
    df['label_simple'] = df['label_simple'].astype(int)
    return df[['id_student', 'code_module', 'code_presentation', 'label_simple']]

def generate_labels_composite(df_feat, df_info, weights, threshold):
    """
    生成复合困难标签
    
    参数:
        df_feat (DataFrame): 特征数据集
        df_info (DataFrame): 学生信息数据
        weights (dict): 各困难因素权重
        threshold (float): 困难阈值(>=该值视为困难)
        
    返回:
        DataFrame: 包含以下列的DataFrame:
            - id_student, code_module, code_presentation: 学生和课程标识
            - difficulty_index: 连续困难指数
            - label_composite: 复合标签(0=正常,1=困难)
    """
    # 合并特征和结果数据
    df = df_feat.merge(
        df_info[['id_student','code_module','code_presentation','final_result']], 
        on=['id_student','code_module','code_presentation'], 
        how='left'
    )
    
    # 1. 准备困难指标组件
    # a. 成绩差(标准化到0-100)
    df['low_score_raw'] = 100 - df['avg_score'].fillna(0)
    # b. 活动量不足(以总点击量代理)
    df['activity_raw'] = df['total_clicks'].fillna(0)
    # c. 活动量差异(班级最大活动量 - 当前学生活动量)
    max_act = df['activity_raw'].max() if not df['activity_raw'].empty else 0
    df['activity_diff_raw'] = max_act - df['activity_raw']
    # d. 作业缺失数
    df['missing_raw'] = df['n_missing'].fillna(0)
    # e. 退学/失败指示器
    df['fail_ind'] = df['final_result'].apply(lambda x: 1 if x in ['Fail','Withdrawn'] else 0)
    
    # 2. 归一化困难指标(0-1范围)
    comp = df[['low_score_raw','activity_diff_raw','missing_raw','fail_ind']].fillna(0)
    scaler = MinMaxScaler()
    norm = scaler.fit_transform(comp)
    norm_df = pd.DataFrame(norm, columns=['low_score_n','activity_diff_n','missing_n','fail_ind_n'], index=df.index)
    
    # 3. 计算综合困难指数
    # 获取权重(默认所有为1.0)
    alpha = weights.get('alpha', 1.0)
    beta = weights.get('beta', 1.0)
    gamma = weights.get('gamma', 1.0)
    delta = weights.get('delta', 1.0)
    
    df['difficulty_index'] = (
        alpha * norm_df['low_score_n'] +
        beta * norm_df['activity_diff_n'] +
        gamma * norm_df['missing_n'] +
        delta * norm_df['fail_ind_n']
    )
    
    # 4. 生成复合标签
    df['label_composite'] = (df['difficulty_index'] >= threshold).astype(int)
    
    return df[['id_student','code_module','code_presentation','difficulty_index','label_composite']]

def build_dataset(data_dir, n_weeks=4, weights=None, threshold=1.5):
    """
    构建完整数据集(特征+标签)
    
    参数:
        data_dir (str): 数据文件目录
        n_weeks (int): 使用的行为周数
        weights (dict): 复合标签权重(默认None,使用等权重)
        threshold (float): 复合标签阈值(默认0.5)
        
    返回:
        DataFrame: 包含特征和标签的完整数据集
        (每行代表一个学生在特定课程学期的学习记录)
        
    返回的数据结构(列说明):
        id_student           # 学生ID(唯一标识)
        code_module          # 课程模块代码
        code_presentation    # 学期表示代码
        
        # 行为特征(前n_weeks)
        total_clicks         # 总点击次数
        active_weeks         # 活跃周数(有活动的周数)
        login_days           # 登录天数(近似)
        clicks_<activity_type> # 各种活动类型的点击次数
        <activity_type>
        'clicks_dataplus', 'clicks_dualpane', 'clicks_externalquiz', 'clicks_forumng', 'clicks_glossary', 'clicks_homepage', 
        'clicks_htmlactivity', 'clicks_oucollaborate', 'clicks_oucontent', 'clicks_ouelluminate', 'clicks_ouwiki', 'clicks_page', 
        'clicks_questionnaire', 'clicks_quiz', 'clicks_resource', 'clicks_sharedsubpage', 'clicks_subpage', 'clicks_url',
        
        # 评估特征
        n_expected_assess    # 课程预期作业数量
        n_attempts           # 学生实际尝试作业数
        n_missing            # 未完成作业数
        avg_score            # 平均分数(0-100)
        
        # 标签(困难定义)
        label_simple         # 简单标签(0=通过,1=不及格/退学)
        difficulty_index     # 复合困难指数(连续值)
        label_composite      # 复合标签(0=正常,1=困难)
    """
    # 1. 加载原始数据
    dfs = load_oulad(data_dir)
    
    # 2. 聚合行为特征(前n_weeks)
    df_feat = aggregate_behavior_features(dfs, n_weeks=n_weeks)
    
    # 3. 获取学生信息
    df_info = dfs['student_info']
    
    # 4. 生成简单标签
    df_label_simple = generate_labels_simple(df_info)
    
    # 5. 生成复合标签
    df_label_comp = generate_labels_composite(
        df_feat, df_info, 
        weights or {},  # 默认空字典(内部使用等权重)
        threshold
    )
    
    # 6. 合并所有数据
    # 首先合并特征和简单标签
    df_all = df_feat.merge(
        df_label_simple, 
        on=['id_student','code_module','code_presentation'], 
        how='left'
    )
    # 然后合并复合标签
    df_all = df_all.merge(
        df_label_comp, 
        on=['id_student','code_module','code_presentation'], 
        how='left'
    )
    
    # 7. 清理数据(移除没有简单标签的行)
    df_all = df_all.dropna(subset=['label_simple'])
    
    return df_all

if __name__ == '__main__':
    # 使用示例
    data_dir = 'anonymisedData'  # 替换为实际数据路径
    weights = {'alpha':1.0, 'beta':0.5, 'gamma':0.5, 'delta':1.0}  # 等权重
    threshold = 1.5  # 复合标签阈值
    
    # 构建数据集
    df = build_dataset(
        data_dir, 
        n_weeks=4, 
        weights=weights, 
        threshold=threshold
    )
    
    # 输出前几行数据
    print("数据集前5行示例:")
    print(df.head())
    
    # 输出数据集基本信息
    print("\n数据集结构信息:")
    print(f"行数: {len(df)}, 列数: {len(df.columns)}")
    print("列名列表:", list(df.columns))
    
    # 输出标签分布
    print("\n标签分布统计:")
    print("简单标签:")
    print(df['label_simple'].value_counts(normalize=True))
    print("\n复合标签:")
    print(df['label_composite'].value_counts(normalize=True))