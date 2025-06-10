import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from read_data import build_dataset
from models import SVMModel

# 配置参数
CONFIG = {
    "data_dir": "anonymisedData",  # 数据目录路径
    "n_weeks": 4,                  # 使用的行为周数
    "weights": {"alpha": 1.0, "beta": 0.5, "gamma": 0.5, "delta": 1.0},  # 复合标签权重
    "composite_threshold": 1.5,     # 复合标签阈值
    "test_size": 0.2,               # 测试集比例
    "val_size": 0.25,               # 验证集比例（从训练集中划分）
    "random_state": 42,             # 随机种子
    "models": {                     # 待训练模型配置
        "LogisticRegression": {"class": None, "params": {}},
        "RandomForest/XGBoost": {"class": None, "params": {"n_estimators": 100}},
        "SVM": {"class": SVMModel, "params": {"probability": True}},
        "MLP": {"class": None, "params": {"hidden_layer_sizes": (128, 64)}},
        "CNN": {"class": None, "params": {}},
        "LSTM": {"class": None, "params": {}}  # 需要特殊数据形状
    },
    "output_dir": "results",        # 输出目录
    "save_models": False,           # 是否保存模型
    "clicks_<activity_type>": 0     # 点击分布是否参与训练
}

def prepare_data(label_type="simple"):
    """
    准备训练数据
    """
    # 加载数据集
    df = build_dataset(
        data_dir=CONFIG["data_dir"],
        n_weeks=CONFIG["n_weeks"],
        weights=CONFIG["weights"],
        threshold=CONFIG["composite_threshold"]
    )
    
    # 根据标签类型选择目标变量
    target_column = "label_simple" if label_type == "simple" else "label_composite"
    print(f"使用标签类型: {label_type}, 正样本比例: {df[target_column].mean():.2%}")
    
    # 准备特征和目标
    X = df.drop(columns=['id_student', 'label_simple', 'label_composite', 'difficulty_index'])
    # 点击分布不参与训练
    if not CONFIG["clicks_<activity_type>"]:
        X.drop(columns=['clicks_dataplus', 'clicks_dualpane', 'clicks_externalquiz', 'clicks_forumng', 'clicks_glossary', 'clicks_homepage', 
        'clicks_htmlactivity', 'clicks_oucollaborate', 'clicks_oucontent', 'clicks_ouelluminate', 'clicks_ouwiki', 'clicks_page', 
        'clicks_questionnaire', 'clicks_quiz', 'clicks_resource', 'clicks_sharedsubpage', 'clicks_subpage', 'clicks_url'])
    y = df[target_column]
    
    # 类别特征编码
    if "code_module" in X.columns or "code_presentation" in X.columns:
        X = pd.get_dummies(X, columns=['code_module', 'code_presentation'])
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG["test_size"],
        stratify=y,
        random_state=CONFIG["random_state"]
    )
    
    
    # 数据标准化 (LSTM除外)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(X.columns),
        "scaler": scaler
    }

def train_model(model_config, data, label_type):
    """
    训练模型

    参数:
        model_config: 模型配置字典
        data: 包含训练、验证和测试数据的字典
        
    返回:
        model: 训练好的模型对象
        metrics: 包含所有评估指标的字典
    """
    model_name = model_config["name"]
    print(f"\n{'='*50}\n训练模型: {model_name}\n{'='*50}")
    
    # 初始化模型
    ModelClass = model_config["class"]
    params = model_config["params"]
    model = ModelClass(**params)
    
    start_time = time.time()
    # 特殊处理序列模型
    if model_name in ["LSTM", "RNN"]:
        # 重塑数据为3D形状 (samples, timesteps, features)
        timesteps = 4  # 假设每周数据作为一个时间步
        X_train_reshaped = reshape_for_rnn(data["X_train"], timesteps)
        X_test_reshaped = reshape_for_rnn(data["X_test"], timesteps)
        
        # 使用统一的fit接口训练模型
        metrics = model.fit(
            X_train_reshaped, data["y_train"],
            X_test_reshaped, data["y_test"],
        )
        train_time = time.time() - start_time
        
    
    else:
        # 使用统一的fit接口训练常规模型
        metrics = model.fit(
            data["X_train"], data["y_train"],
            data["X_test"], data["y_test"],
        )
        train_time = time.time() - start_time
    
    # 合并所有指标
    metrics = {
        "model": model_name,
        "train_time": train_time,
        "metrics": metrics,
    }
    
    # 保存模型
    if CONFIG["save_models"]:
        model_dir = os.path.join(CONFIG["output_dir"], "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_name}_{label_type}.pkl")
        model.save(model_path)
        
        print(f"模型保存至: {model_path}")
    
    return model, metrics

def reshape_for_rnn(X, timesteps):
    """将数据重塑为RNN/LSTM需要的3D形状"""
    num_features = X.shape[1] // timesteps
    return X.reshape((X.shape[0], timesteps, num_features))


def visualize_results(results, label_type):
    """
    可视化模型性能对比
    """
    # 展平结果数据以便可视化
    flat_results = {}
    for model_name, model_data in results.items():
        # 合并外层和内层 metrics
        flat_results[model_name] = {
            "train_time": model_data["train_time"],
            **model_data["metrics"]  # 包含训练和测试指标
        }
    
    # 创建结果数据框
    df_results = pd.DataFrame(flat_results).T
    
    # 1. 训练集性能指标柱状图
    plt.figure(figsize=(15, 8))
    train_metrics = ['train_accuracy', 'train_precision', 'train_recall', 'train_f1']
    
    for i, metric in enumerate(train_metrics):
        plt.subplot(2, 2, i+1)
        sns.barplot(x=df_results.index, y=metric, data=df_results)
        plt.title(f"training performance: {metric.split('_')[1].capitalize()}")
        plt.ylabel(metric.split('_')[1])
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    train_plot_path = os.path.join(CONFIG["output_dir"], f"train_performance_{label_type}.png")
    plt.savefig(train_plot_path)
    plt.close()
    print(f"训练集性能对比图保存至: {train_plot_path}")
    
    # 2. 测试集性能指标柱状图
    plt.figure(figsize=(15, 8))
    test_metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
    
    for i, metric in enumerate(test_metrics):
        plt.subplot(2, 2, i+1)
        sns.barplot(x=df_results.index, y=metric, data=df_results)
        plt.title(f"test performance: {metric.split('_')[1].capitalize()}")
        plt.ylabel(metric.split('_')[1])
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    test_plot_path = os.path.join(CONFIG["output_dir"], f"test_performance_{label_type}.png")
    plt.savefig(test_plot_path)
    plt.close()
    print(f"测试集性能对比图保存至: {test_plot_path}")
    
    # 3. 训练时间对比
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df_results.index, y='train_time', data=df_results)
    plt.title("Models Training Time")
    plt.ylabel("Time (s)")
    plt.xticks(rotation=45)
    
    time_plot_path = os.path.join(CONFIG["output_dir"], f"training_time_{label_type}.png")
    plt.savefig(time_plot_path)
    plt.close()
    print(f"训练时间对比图保存至: {time_plot_path}")

def save_results(results, label_type):
    """保存结果到文件"""
    # 创建输出目录
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # 保存JSON格式结果 - 保持原始嵌套结构
    json_path = os.path.join(CONFIG["output_dir"], f"results_{label_type}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # 展平结果数据以便CSV保存
    flat_results = {}
    for model_name, model_data in results.items():
        flat_results[model_name] = {
            "model": model_name,
            "train_time": model_data["train_time"],
            **model_data["metrics"]
        }
    
    # 保存CSV格式结果
    df = pd.DataFrame(flat_results).T
    csv_path = os.path.join(CONFIG["output_dir"], f"results_{label_type}.csv")
    df.to_csv(csv_path)
    
    print(f"结果保存至: {json_path} 和 {csv_path}")

def main():
    """主训练流程"""
    # 创建输出目录
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # 对两种标签类型进行训练
    for label_type in ["simple", "composite"]:
        print(f"\n{'='*50}\n开始训练 [{label_type.upper()} 标签]\n{'='*50}")
        
        # 准备数据
        data = prepare_data(label_type)
        
        # 存储结果
        results = {}
        
        # 训练所有模型
        for model_name, model_config in CONFIG["models"].items():
            # 跳过未配置的模型
            if model_config["class"] is None:
                continue
                
            model_config["name"] = model_name
            model, metrics = train_model(model_config, data, label_type)
            results[model_name] = metrics
        
        # 保存结果
        save_results(results, label_type)
        
        # 可视化对比
        visualize_results(results, label_type)
    
    print("\n训练完成,所有模型结果已保存")

if __name__ == "__main__":
    main()