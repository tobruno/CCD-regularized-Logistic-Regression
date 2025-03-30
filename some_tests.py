from algorithm_implementation import LogRegCCD
from data_preparation import get_heart_data, get_wine_data
import numpy as np
import matplotlib.pyplot as plt

measures = [
    "recall",
    "precision",
    "F-measure",
    "balanced accuracy",
    "area under the ROC curve",
    "area under the sensitivity-precision curve"
]

X_train, X_test, y_train, y_test = get_wine_data()
alphas = 10 ** np.linspace(start=-8, stop=5, num=500)
models = [LogRegCCD(alpha=alpha) for alpha in alphas]

for model in models:
    model.fit(X_train, y_train)

coefs = [model.coef_ for model in models]
intercepts = [model.intercept_ for model in models]
losses = [model.loss_history[-1] for model in models]

plt.figure(figsize=(10, 6))
plt.plot(alphas, losses, label='Loss')
plt.xscale('log')
plt.xlabel('Regularization strength')
plt.ylabel('Final loss')
plt.title('Final loss against regularization strength')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
for i, coef in enumerate(zip(*coefs)):
    plt.plot(alphas, coef, label=f'Coefficient {i+1}')

plt.plot(alphas, intercepts, label='Intercept', linestyle='--', color='black')
plt.xscale('log')
plt.xlabel('Regularization strength')
plt.ylabel('Coefficient values')
plt.title('Regularization path')
plt.legend()
plt.grid()
plt.show()