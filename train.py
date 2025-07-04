import os
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from read_data import build_dataset
from models import SVMModel, LSTMModel, RandomForestModel, MLPModel, LogisticRegressionModel, RNNModel
from shap_analysis import SHAPAnalyzer

# 配置参数
CONFIG = {
    "data_dir": "anonymisedData",  # 数据目录路径
    "n_weeks": 4,  # 使用的行为周数
    "weights": {"alpha": 1.0, "beta": 0.5, "gamma": 0.5, "delta": 1.0},  # 复合标签权重
    "composite_threshold": 1.5,  # 复合标签阈值
    "test_size": 0.2,  # 测试集比例
    "val_size": 0.25,  # 验证集比例（从训练集中划分）
    "random_state": 42,  # 随机种子
    "models": {  # 待训练模型配置D
        "LogisticRegression": {"class": LogisticRegressionModel, "params": {"max_iter": 200}},
        "RandomForest": {"class": RandomForestModel, "params": {"n_estimators": 300, "max_depth": 15
            , "max_features": "log2", "min_samples_leaf": 4, "min_samples_split": 7, "bootstrap": False,
                                                                "class_weight": "balanced"}},
        "SVM": {"class": SVMModel, "params": {"probability": True}},
        "MLP": {"class": MLPModel, "params": {"hidden_layer_sizes": (72, 248, 212), "batch_size": 128}},
        "RNN": {"class": RNNModel, "params": {"num_epochs": 10, "batch_size": 64, }},
        "LSTM": {"class": LSTMModel, "params": {"num_epochs": 10, "batch_size": 64, }},
    },
    "output_dir": "results",  # 输出目录
    "save_models": False,  # 是否保存模型
    "clicks_<activity_type>": 0,  # 点击分布是否参与训练
    "shap_analysis": True,  # 是否进行SHAP特征分析
    "shap_sample_size": 100,  # 用于SHAP分析的样本数量
    "shap_plot_types": ["bar", "beeswarm"]  # 要生成的SHAP图表类型
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
        X.drop(
            columns=['clicks_dataplus', 'clicks_dualpane', 'clicks_externalquiz', 'clicks_forumng', 'clicks_glossary',
                     'clicks_homepage',
                     'clicks_htmlactivity', 'clicks_oucollaborate', 'clicks_oucontent', 'clicks_ouelluminate',
                     'clicks_ouwiki', 'clicks_page',
                     'clicks_questionnaire', 'clicks_quiz', 'clicks_resource', 'clicks_sharedsubpage', 'clicks_subpage',
                     'clicks_url'])
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
        "X_train": X_train_scaled.values if isinstance(X_train_scaled, pd.DataFrame) else X_train_scaled,
        "X_test": X_test_scaled.values if isinstance(X_test_scaled, pd.DataFrame) else X_test_scaled,
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
    print(f"\n{'=' * 50}\n训练模型: {model_name}\n{'=' * 50}")

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

    # 添加SHAP分析
    if CONFIG.get("shap_analysis", False):
        try:
            print(f"开始SHAP特征分析: {model_name}")
            shap_analyzer = SHAPAnalyzer()

            # 准备数据样本
            sample_size = min(CONFIG.get("shap_sample_size", 100), len(data["X_test"]))
            sample_indices = np.random.choice(len(data["X_test"]), sample_size, replace=False)
            X_sample = data["X_test"][sample_indices]

            # 特殊处理序列模型
            if model_name in ["LSTM", "RNN"]:
                timesteps = 4
                # 确保数据是3D格式
                if len(X_sample.shape) == 2:
                    X_sample = reshape_for_rnn(X_sample, timesteps)
                shap_analyzer.analyze_model(
                    model.model,
                    X_sample,
                    data["feature_names"],
                    model_name,
                    label_type,
                    CONFIG.get("shap_plot_types", ["bar", "beeswarm"])
                )
            else:
                # 确保数据是2D格式
                if len(X_sample.shape) > 2:
                    X_sample = X_sample.reshape(X_sample.shape[0], -1)
                shap_analyzer.analyze_model(
                    model.get_model(),
                    X_sample,
                    data["feature_names"],
                    model_name,
                    label_type,
                    CONFIG.get("shap_plot_types", ["bar", "beeswarm"])
                )
        except Exception as e:
            print(f"SHAP分析出错: {str(e)}")

    return model, metrics


def reshape_for_rnn(X, timesteps):
    """将数据重塑为RNN/LSTM需要的3D形状"""
    num_features = X.shape[1] // timesteps
    return X.reshape((X.shape[0], timesteps, num_features))


def visualize_results(results, label_type):
    """
    可视化模型性能对比
    """

    flat_results = {}
    for model_name, model_data in results.items():
        flat_results[model_name] = {
            "train_time": model_data["train_time"],
            **model_data["metrics"]
        }

    df_results = pd.DataFrame(flat_results).T
    df_results.index.name = 'Model'
    df_results.reset_index(inplace=True)

    palette = sns.color_palette("Set2", n_colors=len(df_results))

    # 1. 训练集性能指标图
    train_metrics = ['train_accuracy', 'train_precision', 'train_recall', 'train_f1']
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(train_metrics):
        plt.subplot(2, 2, i + 1)
        sns.barplot(
            x='Model', y=metric, data=df_results,
            palette=palette
        )
        plt.ylim(0, 1)
        plt.title(f"Train {metric.split('_')[1].capitalize()}", fontsize=12)
        plt.ylabel(metric.split('_')[1])
        plt.xticks(rotation=30)
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        # 添加数值标签
        for idx, val in enumerate(df_results[metric]):
            plt.text(idx, val + 0.01, f"{val:.2f}", ha='center', fontsize=9)

    plt.tight_layout()
    train_plot_path = os.path.join(CONFIG["output_dir"], f"train_performance_{label_type}.png")
    plt.savefig(train_plot_path)
    plt.close()
    print(f"训练集性能对比图保存至: {train_plot_path}")

    # 2. 测试集性能指标图
    test_metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(test_metrics):
        plt.subplot(2, 2, i + 1)
        sns.barplot(
            x='Model', y=metric, data=df_results,
            palette=palette
        )
        plt.ylim(0, 1)
        plt.title(f"Test {metric.split('_')[1].capitalize()}", fontsize=12)
        plt.ylabel(metric.split('_')[1])
        plt.xticks(rotation=30)
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        for idx, val in enumerate(df_results[metric]):
            plt.text(idx, val + 0.01, f"{val:.2f}", ha='center', fontsize=9)

    plt.tight_layout()
    test_plot_path = os.path.join(CONFIG["output_dir"], f"test_performance_{label_type}.png")
    plt.savefig(test_plot_path)
    plt.close()
    print(f"测试集性能对比图保存至: {test_plot_path}")

    # 3. 训练时间对比
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='train_time', data=df_results, palette=palette)
    plt.title("Training Time (Seconds)", fontsize=13)
    plt.ylabel("Time (s)")
    plt.xticks(rotation=30)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for idx, val in enumerate(df_results['train_time']):
        plt.text(idx, val + 0.1, f"{val:.1f}", ha='center', fontsize=9)

    plt.tight_layout()
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
        print(f"\n{'=' * 50}\n开始训练 [{label_type.upper()} 标签]\n{'=' * 50}")

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
