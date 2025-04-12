from algorithm_implementation import LogRegCCD, MultiLambdaLogRegCCD
from data_preparation import get_heart_data, get_wine_data, get_cancer_data, get_synthetic_data,  get_titanic_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score
import pandas as pd

from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    f1_score, balanced_accuracy_score
)

def evaluate_real_data():
    X_train, X_test, y_train, y_test = get_heart_data()
    results = {}
    feature_names = X_train.columns.tolist()

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    proba_lr = lr.predict_proba(X_test)[:, 1]
    pred_lr = (proba_lr >= 0.5).astype(int)

    results['LR_ROC_AUC'] = roc_auc_score(y_test, proba_lr)
    results['LR_PR_AUC'] = average_precision_score(y_test, proba_lr)
    results['LR_F1'] = f1_score(y_test, pred_lr)
    results['LR_BalAcc'] = balanced_accuracy_score(y_test, pred_lr)

    # LogRegCCD
    ccd_lr = LogRegCCD()
    ccd_lr.fit(X_train.values, y_train.values)
    proba_ccd = ccd_lr.predict_proba(X_test.values)
    pred_ccd = (proba_ccd >= 0.5).astype(int)

    lr_coefs = lr.coef_.flatten()
    ccd_coefs = ccd_lr.coef_.flatten()  
    df_coefs = pd.DataFrame({
        'Feature': feature_names,
        'LogisticRegression': lr_coefs,
        'LogRegCCD': ccd_coefs
    })

    results['CCD_ROC_AUC'] = roc_auc_score(y_test, proba_ccd)
    results['CCD_PR_AUC'] = average_precision_score(y_test, proba_ccd)
    results['CCD_F1'] = f1_score(y_test, pred_ccd)
    results['CCD_BalAcc'] = balanced_accuracy_score(y_test, pred_ccd)

    return results, df_coefs


results, df = evaluate_real_data()
df.plot(x='Feature', kind='bar', figsize=(12,6))
plt.title("Coefficient Comparison: LogisticRegression vs LogRegCCD")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


df = pd.DataFrame.from_dict(results, orient='index', columns=['Score'])
df = df.reset_index()
df[['Model', 'Metric']] = df['index'].str.extract(r'(LR|CCD)_(.*)')
df_pivot = df.pivot(index='Metric', columns='Model', values='Score')
df_pivot = df_pivot.loc[['ROC_AUC', 'PR_AUC', 'F1', 'BalAcc']]

df_pivot.plot(kind='bar', figsize=(10, 6))
plt.title("Performance Comparison: LogisticRegression vs LogRegCCD")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0, 1.05)
plt.grid(axis='y')
plt.tight_layout()
plt.show()