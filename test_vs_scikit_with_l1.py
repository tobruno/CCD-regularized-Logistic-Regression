from algorithm_implementation import LogRegCCD, MultiLambdaLogRegCCD
from data_preparation import get_heart_data, get_college_dropout, get_cancer_data, get_synthetic_data,  get_titanic_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score
import pandas as pd

import os
os.makedirs('plots', exist_ok=True)

def evaluate_real_data(X_train, X_test, y_train, y_test ):
    """
    Generate and train models for comparison. The lambda in LogRegCCD is optimised by the balanced accuracy score.

    Parameters:
    -----------
    X_train, X_test, y_train, y_test - training and validating sets and labels
    
    Returns:
    --------
    results by metrices, dataframe with coefficients
    """
    results = {}
    feature_names = X_train.columns.tolist()

    # Logistic Regression
    lr = LogisticRegression(penalty='l1', solver='liblinear')
    lr.fit(X_train, y_train)
    proba_lr = lr.predict_proba(X_test)[:, 1]
    pred_lr = (proba_lr >= 0.5).astype(int)

    results['LR_ROC_AUC'] = roc_auc_score(y_test, proba_lr) 
    results['LR_PR_AUC'] = average_precision_score(y_test, proba_lr)
    results['LR_F1'] = f1_score(y_test, pred_lr)
    results['LR_BalAcc'] = balanced_accuracy_score(y_test, pred_lr)

    # LogRegCCD
    models = MultiLambdaLogRegCCD(start=-8, stop=5, num=500)
    models.fit(X_train, y_train)
    scores = [model.validate(X_test, y_test, "balanced accuracy") for model in models.models]
    # Best AUC ROC score model
    max_score = np.argmax(scores)
    ccd_lr = models.models[max_score]
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


datasets = {'college': get_college_dropout,'cancer':get_cancer_data, 'titanic':get_titanic_data, 'heart': get_heart_data}

for name, dataset in datasets.items():
    X_train, X_test, y_train, y_test = dataset()

    results, df = evaluate_real_data(X_train, X_test, y_train, y_test)
    df.plot(x='Feature', kind='bar', figsize=(12,6))
    plt.title(f"Coefficient Comparison: LogisticRegression(with L1) vs LogRegCCD - {name} dataset")
    plt.ylabel("Coefficient Value")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'plots/l1_comp_{name}_coefs.png')
    plt.show()


    df = pd.DataFrame.from_dict(results, orient='index', columns=['Score'])
    df = df.reset_index()
    df[['Model', 'Metric']] = df['index'].str.extract(r'(LR|CCD)_(.*)')
    df_pivot = df.pivot(index='Metric', columns='Model', values='Score')
    df_pivot = df_pivot.loc[['ROC_AUC', 'PR_AUC', 'F1', 'BalAcc']]

    df_pivot.plot(kind='bar', figsize=(10, 6))
    plt.title(f"Performance Comparison: LogisticRegression(with L1) vs LogRegCCD - {name} dataset")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.ylim(0, 1.05)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f'plots/l1_comp_{name}_evaluation.png')
    plt.show()