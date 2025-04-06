import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from data_preparation import get_heart_data, get_cancer_data, get_synthetic_data, get_titanic_data, get_wine_data

class LogRegCCD:
    def __init__(self, alpha=1.0, max_iter=100, tol=1e-4):
        self.alpha = alpha  # L1 Regularization strength
        self.max_iter = max_iter
        self.tol = tol  # Convergence threshold
        self.coef_ = None  # Model coefficients
        self.intercept_ = 0  # Bias term
        self.loss_history = []

    def sigmoid(self, z):
        z = np.clip(z, -25, 25)
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )

    def predict_proba(self, X):
        return self.sigmoid(X @ self.coef_ + self.intercept_)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def soft_thresholding(self, a, b):
        if a > b:
            return a - b
        elif a < -b:
            return a + b
        else:
            return 0

    def fit(self, X, y):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X  # Convert DataFrame to NumPy array
        y = y.to_numpy() if isinstance(y, pd.Series) else y
        
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)  # Initialize weights
        self.intercept_ = 0
        
        for iteration in range(self.max_iter):
            prev_coef = self.coef_.copy()
            
            for j in range(n_features):  # Coordinate-wise updates
                z = X @ self.coef_ + self.intercept_
                y_pred = self.sigmoid(z)

                residual = y - y_pred
                gradient = np.dot(X[:, j], residual)  # Now X[:, j] is valid
            
                # Soft-thresholding for L1 regularization
                if gradient > self.alpha:
                    self.coef_[j] = (gradient - self.alpha) / np.sum(X[:, j] ** 2)
                elif gradient < -self.alpha:
                    self.coef_[j] = (gradient + self.alpha) / np.sum(X[:, j] ** 2)
                else:
                    self.coef_[j] = 0
            
            self.intercept_ += np.mean(residual)
            
            # Compute loss
            loss = log_loss(y, self.sigmoid(X @ self.coef_ + self.intercept_))
            self.loss_history.append(loss)
            
            if np.linalg.norm(self.coef_ - prev_coef, ord=1) < self.tol:
                break

    def validate(self, X_valid, y_valid, measure):
        measures = {
            "recall": recall_score,
            "precision": precision_score,
            "F-measure": f1_score,
            "balanced accuracy": balanced_accuracy_score,
            "area under the ROC curve": roc_auc_score,
            "area under the sensitivity-precision curve": average_precision_score
        }

        if measure not in measures:
            raise ValueError(
                "Enter either: recall, precision, F-measure, balanced accuracy, area under the ROC curve, or area under the sensitivity-precision curve.")

        X_valid = X_valid.to_numpy() if isinstance(X_valid, pd.DataFrame) else X_valid
        y_valid = y_valid.to_numpy() if isinstance(y_valid, pd.Series) else y_valid

        y_scores = self.predict_proba(X_valid)
        y_pred = (y_scores >= 0.5).astype(int)

        if measure in ["area under the ROC curve", "area under the sensitivity-precision curve"]:
            metric_input = y_scores # use scores if area under the curve metric
        else:
            metric_input = y_pred # use binary prediction otherwise

        return measures[measure](y_valid, metric_input)
    
    def plot_loss(self):
        plt.plot(self.loss_history, label='Loss Convergence')
        plt.xlabel('Iterations')
        plt.ylabel('Log-Loss')
        plt.title('Loss Convergence Over Iterations')
        plt.legend()
        plt.show()


datasets = {'wine': get_wine_data,'cancer':get_cancer_data, 'titanic':get_titanic_data, 'heart': get_heart_data, 'synthetic': get_synthetic_data}

# for dataset in datasets.values():
#     print(dataset)
#     X_train, X_test, y_train, y_test = dataset()
#     model = LogRegCCD(alpha=0.1, max_iter=100)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     print('Dataset:', dataset)
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     print("Precision:", precision_score(y_test, y_pred))
#     print("Recall:", recall_score(y_test, y_pred))
#     print("F1-score:", f1_score(y_test, y_pred))
#     model.plot_loss()

# Evaluation Metrics
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Precision:", precision_score(y_test, y_pred))
# print("Recall:", recall_score(y_test, y_pred))
# print("F1-score:", f1_score(y_test, y_pred))
# model.plot_loss()
