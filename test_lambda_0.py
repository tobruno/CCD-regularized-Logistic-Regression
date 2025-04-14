from algorithm_implementation import LogRegCCD
from data_preparation import get_heart_data, get_college_dropout, get_cancer_data, get_synthetic_data, get_titanic_data
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score
import pandas as pd

def evaluate_metrics(model, X_test, y_test):
    X_test_np = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test
    y_test_np = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test
    
    y_proba = model.predict_proba(X_test_np)
    y_pred = (y_proba >= 0.5).astype(int)
    
    metrics = {
        "Accuracy": accuracy_score(y_test_np, y_pred),
        "Recall": recall_score(y_test_np, y_pred),
        "Precision": precision_score(y_test_np, y_pred),
        "F-measure": f1_score(y_test_np, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test_np, y_pred),
        "ROC AUC": roc_auc_score(y_test_np, y_proba),
        "PR AUC": average_precision_score(y_test_np, y_proba),
        "Log Loss": log_loss(y_test_np, y_proba)
    }
    
    return metrics

# Test datasets
datasets = {
    'Cancer': get_cancer_data,
    'Heart': get_heart_data,
    'College': get_college_dropout,
    'Titanic': get_titanic_data,
    'Synthetic': get_synthetic_data
}

results = []

for name, dataset_func in datasets.items():
    X_train, X_test, y_train, y_test = dataset_func()
    
    model = LogRegCCD(lambda_param=0.0, max_iter=200)
    model.fit(X_train, y_train)
    
    model.plot_loss(filename=f'plots/loss_{name}.png')
    model.plot_coef_history(filename=f'plots/coef_{name}.png')

    metrics = evaluate_metrics(model, X_test, y_test)
    metrics['Dataset'] = name
    results.append(metrics)
    
    final_loss = model.loss_history[-1] if model.loss_history else "N/A"

results_df = pd.DataFrame(results)
cols = ['Dataset'] + [col for col in results_df.columns if col != 'Dataset']
results_df = results_df[cols]

results_df.to_csv('lambda_zero_performance.csv', index=False)
print("\nResults saved to lambda_zero_performance.csv")