from algorithm_implementation import LogRegCCD, MultiLambdaLogRegCCD
from data_preparation import get_heart_data, get_college_dropout, get_cancer_data, get_synthetic_data,  get_titanic_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score
import pandas as pd

def validate(model, X_valid, y_valid, measure):
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

    # X_valid = X_valid.to_numpy() if isinstance(X_valid, pd.DataFrame) else X_valid
    # y_valid = y_valid.to_numpy() if isinstance(y_valid, pd.Series) else y_valid

    y_scores = model.predict_proba(X_valid)[ : ,1]
    y_pred = model.predict(X_valid)

    if measure in ["area under the ROC curve", "area under the sensitivity-precision curve"]:
        metric_input = y_scores # use scores if area under the curve metric
        #print(metric_input)
    else:
        metric_input = y_pred # use binary prediction otherwise

    return measures[measure](y_valid, metric_input)

measures = [
    "recall",
    "precision",
    "F-measure",
    "balanced accuracy",
    "area under the ROC curve",
    "area under the sensitivity-precision curve"
]

prompt = "Choose one from the following:\n 0 - recall,\n 1 - precision, \n 2 - F-measure, \n 3 - balanced accuracy, \n 4 - area under the ROC curve, \n 5 - area under the sensitivity-precision curve\n: "

number_of_measure = int(input(prompt))
measure = measures[number_of_measure]

X_train, X_valid, y_train, y_valid = get_synthetic_data()
models = MultiLambdaLogRegCCD(start=-8, stop=5, num=500)
models.fit(X_train, y_train)

models.plot_final_loss()
models.plot_coefficients()
models.plot(measure, X_valid, y_valid)


# max_score = np.argmax(scores)
# model = models[max_score]
#
# print(f"\nChosen model: \nlambda: {alphas[max_score]},\nloss: {losses[max_score]},\n{measure} score: {scores[max_score]},\ncoefficients: {coefs[max_score]}")
#
# sklearn_model = LogisticRegression(penalty=None)
# sklearn_model.fit(X_train, y_train)
# score = validate(sklearn_model, X_valid=X_valid, y_valid=y_valid, measure=measure)
# print(f"\nSklearn's Logistic Regression model:\n{measure} score: {score}")


