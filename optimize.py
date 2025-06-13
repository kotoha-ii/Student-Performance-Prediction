import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from models import SVMModel, LSTMModel, MLPModel, RandomForestModel
from read_data import build_dataset

CONFIG = {
    "data_dir": "anonymisedData",
    "n_weeks": 4,
    "weights": {"alpha": 1.0, "beta": 0.5, "gamma": 0.5, "delta": 1.0},
    "threshold": 1.5,
    "label_type": "simple",  # 可切换为 "composite"
    "test_size": 0.2,
    "random_state": 42,
    "timesteps": 4
}

def prepare_data(label_type):
    df = build_dataset(CONFIG["data_dir"], CONFIG["n_weeks"], CONFIG["weights"], CONFIG["threshold"])
    y = df["label_simple"] if label_type == "simple" else df["label_composite"]
    X = df.drop(columns=["id_student", "label_simple", "label_composite", "difficulty_index"], errors="ignore")
    X = pd.get_dummies(X, columns=["code_module", "code_presentation"])
    return train_test_split(X, y, test_size=CONFIG["test_size"], stratify=y, random_state=CONFIG["random_state"])

def optimize_svm(X_train, X_val, y_train, y_val):
    def objective(trial):
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
        C = trial.suggest_float("C", 1e-2, 1e2, log=True)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        model = SVMModel(kernel=kernel, C=C, gamma=gamma, probability=True)
        metrics = model.fit(X_train, y_train, X_val, y_val)
        return metrics["test_f1"]

    study = optuna.create_study(direction="maximize", study_name="SVM Optimization")
    study.optimize(objective, n_trials=30)
    print("最佳 SVM 参数:", study.best_params)
    print("最佳 F1 分数:", study.best_value)
    study.trials_dataframe().to_csv(f"svm_optuna_results_{CONFIG['label_type']}.csv", index=False)

def optimize_lstm(X_train, X_val, y_train, y_val):
    def objective(trial):
        hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        num_epochs = 10
        num_features = X_train.shape[1] // CONFIG["timesteps"]
        X_train_r = X_train.values.reshape(-1, CONFIG["timesteps"], num_features)
        X_val_r = X_val.values.reshape(-1, CONFIG["timesteps"], num_features)
        model = LSTMModel(input_dim=num_features, hidden_dim=hidden_dim, lr=lr, batch_size=batch_size, num_epochs=num_epochs)
        metrics = model.fit(X_train_r, y_train, X_val_r, y_val)
        return metrics["test_f1"]

    study = optuna.create_study(direction="maximize", study_name="LSTM Optimization")
    study.optimize(objective, n_trials=20)
    print("最佳 LSTM 参数:", study.best_params)
    print("最佳 F1 分数:", study.best_value)
    study.trials_dataframe().to_csv(f"lstm_optuna_results_{CONFIG['label_type']}.csv", index=False)


def optimize_random_forest(X_train, X_val, y_train, y_val):
    def objective(trial):
        # 参数搜索空间
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }

        # 初始化模型
        model = RandomForestModel(**params)

        # 训练与评估
        metrics = model.fit(X_train, y_train, X_val, y_val)
        return metrics["test_f1"]

    # 创建Optuna研究
    study = optuna.create_study(direction="maximize", study_name="RF_Optimization")
    study.optimize(objective, n_trials=50)

    # 输出结果
    print("最佳随机森林参数:", study.best_params)
    print("最佳F1分数:", study.best_value)
    study.trials_dataframe().to_csv(f"rf_optuna_results_{CONFIG['label_type']}.csv", index=False)


def optimize_mlp(X_train, X_val, y_train, y_val):
    # 确保输入是 NumPy 数组而不是 DataFrame
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(X_val, pd.DataFrame):
        X_val = X_val.values
    # if isinstance(y_train, (pd.DataFrame, pd.Series)):
    #     y_train = y_train.values
    # if isinstance(y_val, (pd.DataFrame, pd.Series)):
    #     y_val = y_val.values

    # 确保数据类型正确
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    # y_train = y_train.astype(np.float32)
    # y_val = y_val.astype(np.float32)

    def objective(trial):
        # 参数搜索空间
        params = {
            'hidden_layer_sizes': tuple([
                trial.suggest_int(f'n_units_layer_{i}', 32, 256)
                for i in range(trial.suggest_int('n_layers', 1, 4))
            ]),
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        }

        # 初始化模型
        model = MLPModel(
            hidden_layer_sizes=params['hidden_layer_sizes'],
            lr=params['lr'],
            batch_size=params['batch_size'],
            num_epochs=20  # 固定训练轮次
        )

        # 训练与评估
        metrics = model.fit(X_train, y_train, X_val, y_val)
        return metrics["test_f1"]

    study = optuna.create_study(direction="maximize", study_name="MLP_Optimization")
    study.optimize(objective, n_trials=30)

    ######################################
    # 新增：结构化输出最佳参数
    ######################################
    print("\n=== 最佳参数组合 ===")
    best_params = study.best_params
    # 格式化隐藏层结构
    hidden_layers = [v for k, v in best_params.items() if k.startswith('n_units_layer')]
    print(f"隐藏层结构：{tuple(hidden_layers)}层神经网络")
    print(f"具体维度：{hidden_layers}")
    print(f"学习率：{best_params['lr']:.2e}")
    print(f"批大小：{best_params['batch_size']}")
    print(f"Dropout比例：{best_params['dropout_rate']:.2f}")
    print(f"权重衰减：{best_params['weight_decay']:.2e}")
    print(f"验证集F1分数：{study.best_value:.4f}")

def main():
    print(f"开始超参数优化任务（标签类型：{CONFIG['label_type']}）")
    X_train, X_val, y_train, y_val = prepare_data(CONFIG["label_type"])
    print("开始优化 SVM")
    # optimize_svm(X_train, X_val, y_train, y_val)
    print("\n开始优化 LSTM")
    # optimize_lstm(X_train, X_val, y_train, y_val)
    print("\n开始优化MLP")
    optimize_mlp(X_train, X_val, y_train, y_val)
    print("\n开始优化 RandForest")
    optimize_random_forest(X_train, X_val, y_train, y_val)
    print("\n所有模型优化完成")

if __name__ == "__main__":
    main()
