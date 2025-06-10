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

### a. Logistic Regression

### b. Random Forest / 梯度提升树（如 XGBoost、LightGBM、CatBoost）

### c. SVM

### d. MLP

### e. LSTM

### f. RNN

## 3. 添加模型

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
    "models": {                     # 待训练模型配置
        "LogisticRegression": {"class": None, "params": {}},
        "RandomForest/XGBoost": {"class": None, "params": {"n_estimators": 100}},
        "SVM": {"class": SVMModel, "params": {"probability": True}},
        "MLP": {"class": None, "params": {"hidden_layer_sizes": (128, 64)}},
        "CNN": {"class": None, "params": {}},
        "LSTM": {"class": None, "params": {}}  # 需要特殊数据形状
    },
```

随后运行train.py即可开始训练并查看结果
