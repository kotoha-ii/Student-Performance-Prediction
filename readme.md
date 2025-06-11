# 学业困难学生识别

利用学生在学习平台上前几周的匿名行为数据（例如：课程视频观看时长、论坛发帖数、测验尝试次数、特定章节平均分、登录频率等信息），构建并对比不同的分类模型，以识别可能存在学业困难（如课程不及格、辍学）的学生。

**Open University Learning Analytics Dataset (OULAD):** 包含学生人口统计信息、与虚拟学习环境的互动以及最终成绩。

## 1. 项目运行

创建环境

```
conda create -n yourname python=3.10
pip install requirements.txt
```

开始训练

```
python train.py
```

## 2. 选用模型

 **models.py**

+ a. Logistic Regression
+ b. Random Forest / 梯度提升树（如 XGBoost、LightGBM、CatBoost）
+ c. SVM
+ d. MLP
+ e. RNN
+ f. LSTM

## 3. 添加模型

+ 我们的框架支持添加您的自定义模型

在models.py实现你的模型 模型必须要有一个fit函数 格式如下

+ 接受 **训练数据集** 和 **测试数据集**
+ 返回 **训练结果** 和 **测试结果** 需要有这八个数据：

results['train_accuracy'] = acc_train
results['train_precision'] = precision
results['train_recall'] = recall
results['train_f1'] = f1

results['test_accuracy'] = acc_test
results['test_precision'] = precision
results['test_recall'] = recall
results['test_f1'] = f1

请参考已经实现的SVM：

```
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the SVM model. Optionally evaluate on a validation set.
        Args:
            X_train: array-like, training features
            y_train: array-like, training labels
            X_val: array-like or None, validation features
            y_val: array-like or None, validation labels
        Returns:
            dict: training metrics; if validation provided, includes validation metrics
        """
        self.logger.info("Starting training...")
        self.model.fit(X_train, y_train)
        self.logger.info("Training completed.")
        results = {}
        # Training metrics
        y_pred_train = self.model.predict(X_train)
        acc_train = accuracy_score(y_train, y_pred_train)
        precision, recall, f1, _ = precision_recall_fscore_support(y_train, y_pred_train, average='binary')
        results['train_accuracy'] = acc_train
        results['train_precision'] = precision
        results['train_recall'] = recall
        results['train_f1'] = f1
        self.logger.info(f"Train metrics - Acc: {acc_train:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # Validation metrics
        if X_val is not None and y_val is not None:
            self.logger.info("Evaluating on validation set...")
            val_metrics = self.evaluate(X_val, y_val)
            results.update(val_metrics)
        return results
```

随后在train.py的CONFIG中注册你的模型

+ class为你的模型类名称
+ params为初始化你的模型类所需的参数

同样参考已实现的SVM

```
    "models": {                     # 待训练模型配置D
        "LogisticRegression": {"class": None, "params": {}},
        "RandomForest/XGBoost": {"class": None, "params": {"n_estimators": 100}},
        "SVM": {"class": SVMModel, "params": {"probability": True}},
        "MLP": {"class": None, "params": {"hidden_layer_sizes": (128, 64)}},
        "CNN": {"class": None, "params": {}},
        "LSTM": {"class": LSTMModel, "params": {"num_epochs": 100, "batch_size": 64,}},
    },
```

随后运行train.py即可开始训练并查看结果

## 4. 参数调优

在optimize.py中我们使用optuna进行自动参数调优 您可以通过如下方式进行模型的参数调优

+ 添加您的模型调优函数

```
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
```

+ 调用并获取最佳参数

```
X_train, X_val, y_train, y_val = prepare_data(CONFIG["label_type"])
print("\n开始优化 LSTM")
optimize_lstm(X_train, X_val, y_train, y_val)
```
