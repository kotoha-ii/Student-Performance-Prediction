import os
import logging
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression


class SVMModel:
    """
    SVMModel encapsulates a Support Vector Machine classifier with training, evaluation, 
    and model persistence functionalities.

    Usage:
        model = SVMModel(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42, log_dir='logs')
        model.fit(X_train, y_train, X_val, y_val)
        metrics = model.evaluate(X_test, y_test)
        model.save('svm_model.pkl')
        model.load('svm_model.pkl')
    """

    def __init__(self, kernel='rbf', C=1.0, gamma='scale', probability=True,
                 random_state=None, log_dir='logs'):
        # Initialize logging
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'svm_model.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Initialize SVM classifier
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=probability, random_state=random_state)
        self.logger.info(f"Initialized SVM with kernel={kernel}, C={C}, gamma={gamma}, probability={probability}")

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

    def predict(self, X):
        """Predict labels for given features."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict probability estimates for given features. Requires probability=True."""
        try:
            return self.model.predict_proba(X)
        except Exception as e:
            self.logger.error("Probability estimates not available. Ensure probability=True when initializing SVC.")
            raise e

    def evaluate(self, X, y, prefix='test'):
        """
        Evaluate the model on given data and return metrics. Also plot ROC curve.
        Args:
            X: array-like, features
            y: array-like, true labels
            prefix: str, prefix for metric keys, e.g., 'test' or 'val'
        Returns:
            dict: metrics including accuracy, precision, recall, f1, auc
        """
        y_pred = self.model.predict(X)
        metrics = {}
        acc = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
        metrics[f'{prefix}_accuracy'] = acc
        metrics[f'{prefix}_precision'] = precision
        metrics[f'{prefix}_recall'] = recall
        metrics[f'{prefix}_f1'] = f1
        self.logger.info(f"{prefix.capitalize()} metrics - Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # ROC AUC if probabilities available
        try:
            proba = self.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, proba)
            metrics[f'{prefix}_auc'] = auc
            self.logger.info(f"{prefix.capitalize()} AUC: {auc:.4f}")
            # Plot ROC curve
            fpr, tpr, _ = roc_curve(y, proba)
            plt.figure()
            plt.plot(fpr, tpr, label=f'{prefix} ROC (AUC = {auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{prefix.capitalize()} ROC Curve')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.tight_layout()
            plot_path = f"results/{prefix}_roc_curve.png"
            plt.savefig(plot_path)
            plt.close()
            self.logger.info(f"Saved ROC curve to {plot_path}")
        except Exception:
            self.logger.warning("Skipping AUC and ROC plot (probability estimates unavailable)")
        return metrics

    def save(self, path):
        """Save the trained model to a file."""
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        joblib.dump(self.model, path)
        self.logger.info(f"Model saved to {path}")

    def load(self, path):
        """Load a model from file."""
        self.model = joblib.load(path)
        self.logger.info(f"Model loaded from {path}")


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.3):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # 取最后一层 hidden state
        out = self.fc(hn[-1])
        return torch.sigmoid(out)

class LSTMModel:
    def __init__(self, input_dim=None, hidden_dim=64, lr=1e-3, num_epochs=10, batch_size=64, log_dir="logs", device=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

        if X_val is not None:
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

        if self.input_dim is None:
            self.input_dim = X_train.shape[2]

        self.model = LSTMNet(self.input_dim, self.hidden_dim).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            for i in range(0, len(X_train), self.batch_size):
                xb = X_train[i:i+self.batch_size].to(self.device)
                yb = y_train[i:i+self.batch_size].to(self.device)

                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss:.4f}")

        # 评估
        return self.evaluate(X_train, y_train, X_val, y_val)

    def evaluate(self, X_train, y_train, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            def predict(X):
                probs = self.model(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().numpy().flatten()
                return (probs >= 0.5).astype(int), probs

            y_pred_train, prob_train = predict(X_train)
            y_pred_test, prob_test = predict(X_test)

        metrics = {
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "train_precision": precision_score(y_train, y_pred_train),
            "train_recall": recall_score(y_train, y_pred_train),
            "train_f1": f1_score(y_train, y_pred_train),
            "train_auc": roc_auc_score(y_train, prob_train),

            "test_accuracy": accuracy_score(y_test, y_pred_test),
            "test_precision": precision_score(y_test, y_pred_test),
            "test_recall": recall_score(y_test, y_pred_test),
            "test_f1": f1_score(y_test, y_pred_test),
            "test_auc": roc_auc_score(y_test, prob_test),
        }

        return metrics

    def save(self, path):
        torch.save(self.model.state_dict(), path + ".pt")
        with open(path + "_meta.pkl", "wb") as f:
            joblib.dump({"input_dim": self.input_dim}, f)

    def load(self, path):
        with open(path + "_meta.pkl", "rb") as f:
            meta = joblib.load(f)
        self.input_dim = meta["input_dim"]
        self.model = LSTMNet(self.input_dim, self.hidden_dim)
        self.model.load_state_dict(torch.load(path + ".pt"))
        self.model.to(self.device)
        self.model.eval()


class RandomForestModel:
    def __init__(self, logger=None, log_dir='logs', **params):
        os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger('RandomForest')
        self.logger.setLevel(logging.INFO)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # 为随机森林创建专用日志文件
        rf_log_path = os.path.join(log_dir, 'random_forest.log')
        file_handler = logging.FileHandler(rf_log_path)
        stream_handler = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        self.logger.info("=== Initializing Random Forest Model ===")
        self.model = RandomForestClassifier(**params)
        self.logger.info(f"Initialized RandomForest with params: {params}")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.logger.info("Starting training...")
        self.model.fit(X_train, y_train)
        self.logger.info("Training completed.")

        results = {}
        # 训练集指标
        y_pred_train = self.model.predict(X_train)
        self._calculate_metrics(y_train, y_pred_train, results, 'train')

        # 验证集指标
        if X_val is not None and y_val is not None:
            self.logger.info("Evaluating on validation set...")
            y_pred_val = self.model.predict(X_val)
            self._calculate_metrics(y_val, y_pred_val, results, 'test')
            try:
                proba_val = self.model.predict_proba(X_val)[:, 1]
                self._calculate_auc(y_val, proba_val, results, 'test')
            except Exception as e:
                self.logger.warning(f"AUC calculation failed: {str(e)}")

        return results

    def _calculate_metrics(self, y_true, y_pred, results, prefix):
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        results.update({
            f'{prefix}_accuracy': acc,
            f'{prefix}_precision': precision,
            f'{prefix}_recall': recall,
            f'{prefix}_f1': f1
        })
        self.logger.info(
            f"{prefix.capitalize()} metrics - Acc: {acc:.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )

    def _calculate_auc(self, y_true, proba, results, prefix):
        auc = roc_auc_score(y_true, proba)
        results[f'{prefix}_auc'] = auc
        self.logger.info(f"{prefix.capitalize()} AUC: {auc:.4f}")
        # 绘制ROC曲线
        fpr, tpr, _ = roc_curve(y_true, proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'{prefix} ROC (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{prefix.capitalize()} ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/{prefix}_roc_curve.png")
        plt.close()

    def save(self, path):
        joblib.dump(self.model, path)
        self.logger.info(f"Model saved to {path}")

    def load(self, path):
        self.model = joblib.load(path)
        self.logger.info(f"Model loaded from {path}")


class MLPModel:
    def __init__(self, input_dim=None, hidden_layer_sizes=(128, 64), lr=0.001,
                 num_epochs=20, batch_size=64, log_dir='logs', device=None):
        self.input_dim = input_dim
        self.hidden_dims = hidden_layer_sizes
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 创建模型特定的日志器
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger('MLP')
        self.logger.setLevel(logging.INFO)

        # 清除现有处理器（防止重复写入）
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # 创建文件处理器
        mlp_log_path = os.path.join(log_dir, 'mlp_model.log')
        file_handler = logging.FileHandler(mlp_log_path)
        stream_handler = logging.StreamHandler()

        # 设置格式
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        self.logger.info("=== Initializing MLP Model ===")

        if input_dim:
            self._init_model(input_dim)

    def _init_model(self, input_dim):
        layers = []
        prev_dim = input_dim
        for dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers).to(self.device)
        self.logger.info(f"Initialized MLP with architecture: {self.model}")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # 数据转换
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

        if self.input_dim is None:
            self.input_dim = X_train.shape[1]
            self._init_model(self.input_dim)

        # 初始化训练组件
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 训练循环
        self.logger.info("Starting training...")
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            for i in range(0, len(X_train), self.batch_size):
                batch_x = X_train[i:i + self.batch_size].to(self.device)
                batch_y = y_train[i:i + self.batch_size].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs} | Loss: {total_loss / len(X_train):.4f}")

        # 评估指标
        results = self._evaluate(X_train, y_train, X_val, y_val)
        return results

    def _evaluate(self, X_train, y_train, X_val, y_val):
        self.model.eval()
        results = {}

        with torch.no_grad():
            # 训练集评估
            y_pred_train = self._predict(X_train)
            self._calc_metrics(y_train.numpy(), y_pred_train, results, 'train')

            # 验证集评估
            if X_val is not None and y_val is not None:
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
                y_pred_val = self._predict(X_val_tensor)
                self._calc_metrics(y_val_tensor.numpy(), y_pred_val, results, 'test')

        return results

    def _predict(self, X):
        outputs = self.model(X.to(self.device))
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        return (probs >= 0.5).astype(int)

    def _calc_metrics(self, y_true, y_pred, results, prefix):
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        results.update({
            f'{prefix}_accuracy': acc,
            f'{prefix}_precision': precision,
            f'{prefix}_recall': recall,
            f'{prefix}_f1': f1
        })
        self.logger.info(
            f"{prefix.capitalize()} Metrics - "
            f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, F1: {f1:.4f}"
        )

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model saved to {path}")

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        self.logger.info(f"Model loaded from {path}")


class LogisticRegressionModel:
    def __init__(self, penalty='l2', C=1.0, solver='lbfgs', max_iter=200, log_dir="logs"):
        self.model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter)
        self.log_dir = log_dir

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        return self.evaluate(X_train, y_train, X_val, y_val)

    def evaluate(self, X_train, y_train, X_test, y_test):
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        y_prob_train = self.model.predict_proba(X_train)[:, 1]
        y_prob_test = self.model.predict_proba(X_test)[:, 1]
        metrics = {
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "train_precision": precision_score(y_train, y_pred_train),
            "train_recall": recall_score(y_train, y_pred_train),
            "train_f1": f1_score(y_train, y_pred_train),
            "train_auc": roc_auc_score(y_train, y_prob_train),
            "test_accuracy": accuracy_score(y_test, y_pred_test),
            "test_precision": precision_score(y_test, y_pred_test),
            "test_recall": recall_score(y_test, y_pred_test),
            "test_f1": f1_score(y_test, y_pred_test),
            "test_auc": roc_auc_score(y_test, y_prob_test)
        }
        return metrics

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

class RNNNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.3):
        super(RNNNet, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, hn = self.rnn(x)
        out = self.fc(hn[-1])
        return torch.sigmoid(out)

class RNNModel:
    def __init__(self, input_dim=None, hidden_dim=64, lr=1e-3, num_epochs=10, batch_size=64, log_dir="logs", device=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

        if X_val is not None:
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

        if self.input_dim is None:
            self.input_dim = X_train.shape[2]

        self.model = RNNNet(self.input_dim, self.hidden_dim).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            for i in range(0, len(X_train), self.batch_size):
                xb = X_train[i:i+self.batch_size].to(self.device)
                yb = y_train[i:i+self.batch_size].to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss:.4f}")

        return self.evaluate(X_train, y_train, X_val, y_val)

    def evaluate(self, X_train, y_train, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            def predict(X):
                probs = self.model(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().numpy().flatten()
                return (probs >= 0.5).astype(int), probs

            y_pred_train, prob_train = predict(X_train)
            y_pred_test, prob_test = predict(X_test)

        metrics = {
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "train_precision": precision_score(y_train, y_pred_train),
            "train_recall": recall_score(y_train, y_pred_train),
            "train_f1": f1_score(y_train, y_pred_train),
            "train_auc": roc_auc_score(y_train, prob_train),

            "test_accuracy": accuracy_score(y_test, y_pred_test),
            "test_precision": precision_score(y_test, y_pred_test),
            "test_recall": recall_score(y_test, y_pred_test),
            "test_f1": f1_score(y_test, y_pred_test),
            "test_auc": roc_auc_score(y_test, prob_test),
        }

        return metrics

    def save(self, path):
        torch.save(self.model.state_dict(), path + ".pt")
        with open(path + "_meta.pkl", "wb") as f:
            joblib.dump({"input_dim": self.input_dim}, f)

    def load(self, path):
        with open(path + "_meta.pkl", "rb") as f:
            meta = joblib.load(f)
        self.input_dim = meta["input_dim"]
        self.model = RNNNet(self.input_dim, self.hidden_dim)
        self.model.load_state_dict(torch.load(path + ".pt"))
        self.model.to(self.device)
        self.model.eval()


