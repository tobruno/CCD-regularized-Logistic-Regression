from algorithm_implementation import LogRegCCD, MultiLambdaLogRegCCD
from data_preparation import get_heart_data, get_wine_data, get_cancer_data, get_synthetic_data,  get_titanic_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score
import pandas as pd


n_vals = [100, 500, 1000, 5000]
p_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
d_vals = [5, 20, 40, 60, 80]
g_vals = [0.1, 0.4, 0.6, 0.9, 1]


def evaluate_models(n=1000, p=0.5, d=20, g=0.5, seed=42):
    X_train, X_test, y_train, y_test = get_synthetic_data(n, p, d, g, seed)

    # LogisticRegression
    lr = LogisticRegression(penalty=None)
    lr.fit(X_train, y_train)
    y_proba_lr = lr.predict_proba(X_test)[:, 1]
    y_pred_lr = (y_proba_lr >= 0.5).astype(int)
    
    auc_lr = roc_auc_score(y_test, y_proba_lr)
    bal_acc_lr = balanced_accuracy_score(y_test, y_pred_lr)
    
    models = MultiLambdaLogRegCCD(start=-8, stop=5, num=500)
    models.fit(X_train, y_train)
    scores = [model.validate(X_test, y_test, "area under the ROC curve") for model in models.models]
    # Best AUC ROC score model
    max_score = np.argmax(scores)
    ccd_lr = models.models[max_score]

    ccd_lr.fit(X_train.values, y_train.values)
    y_proba_ccd = ccd_lr.predict_proba(X_test.values)
    y_pred_ccd = (y_proba_ccd >= 0.5).astype(int)

    auc_ccd = roc_auc_score(y_test, y_proba_ccd)
    bal_acc_ccd = balanced_accuracy_score(y_test, y_pred_ccd)

    return {
        'ROC_AUC_LR': auc_lr,
        'Balanced_Acc_LR': bal_acc_lr,
        'ROC_AUC_CCD': auc_ccd,
        'Balanced_Acc_CCD': bal_acc_ccd
    }

param_grid = {
    'n': n_vals,
    'p': p_vals,
    'd': d_vals,
    'g': g_vals,
}

def run_param_sweep(vary='n', fixed_vals=None):
    if fixed_vals is None:
        fixed_vals = {'n': 1000, 'p': 0.5, 'd': 20, 'g': 0.5}

    results = []

    for val in param_grid[vary]:
        params = fixed_vals.copy()
        params[vary] = val
        scores = evaluate_models(**params)
        scores[vary] = val
        results.append(scores)

    df = pd.DataFrame(results)
    return df

def plot_results(df, param_name):
    plt.figure(figsize=(10, 6))

    plt.plot(df[param_name], df['ROC_AUC_LR'], label='ROC AUC - LogisticRegression', marker='o')
    plt.plot(df[param_name], df['ROC_AUC_CCD'], label='ROC AUC - LogRegCCD', marker='o')
    plt.plot(df[param_name], df['Balanced_Acc_LR'], label='Balanced Acc - LogisticRegression', marker='s')
    plt.plot(df[param_name], df['Balanced_Acc_CCD'], label='Balanced Acc - LogRegCCD', marker='s')

    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.title(f'Model Performance vs {param_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/task3_synthetic_{param_name}_coefs.png')
    plt.show()

df_n = run_param_sweep(vary='n')
plot_results(df_n, 'n')
df_p = run_param_sweep(vary='p')
plot_results(df_p, 'p')
df_d = run_param_sweep(vary='d')
plot_results(df_d, 'd')
df_g = run_param_sweep(vary='g')
plot_results(df_g, 'g')