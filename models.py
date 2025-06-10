import os
import logging
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve


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
