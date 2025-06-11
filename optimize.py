import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from models import SVMModel, LSTMModel
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

def main():
    print(f"开始超参数优化任务（标签类型：{CONFIG['label_type']}）")
    X_train, X_val, y_train, y_val = prepare_data(CONFIG["label_type"])
    print("开始优化 SVM")
    optimize_svm(X_train, X_val, y_train, y_val)
    print("\n开始优化 LSTM")
    optimize_lstm(X_train, X_val, y_train, y_val)
    print("\n所有模型优化完成")

if __name__ == "__main__":
    main()
