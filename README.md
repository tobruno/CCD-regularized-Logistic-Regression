# CCD-regularized Logistic Regression

This project focuses on implementing the Cyclic Coordinate Descent (CCD) algorithm for L1-regularized logistic regression.

## Project Structure

- `algorithm_implementation.py`: Contains the implementation of the LogRegCCD and MultiLambdaLogRegCCD classes.
- `data_preparation.py`: Functions for preparing and loading datasets.
- `some_tests.py`: Basic tests for the LogRegCCD algorithm.
- `test_lambda_0.py`: Tests for LogRegCCD with lambda=0 to verify correctness.
- `test_vs_scikit_with_l1.py`: Compares LogRegCCD with scikit-learn's LogisticRegression with L1 penalty.
- `task_3_1.py`: Analyzes how synthetic data parameters affect model performance.
- `task_3_2.py`: Compares LogRegCCD and standard LogisticRegression on real datasets.

## Dataset Preparation (Task 1)

The project uses both real-world datasets and synthetic data:

- **Real datasets**: Cancer, Heart Disease, College Dropout, and Titanic
- **Synthetic dataset**: Generated according to the specifications from Task 1

All datasets are processed in `data_preparation.py`.

## Implementation of LogRegCCD (Task 2)

The CCD algorithm is implemented in `algorithm_implementation.py`. The main classes are:

- `LogRegCCD`: Implements the core CCD algorithm for a single lambda value
- `MultiLambdaLogRegCCD`: Runs the algorithm for multiple lambda values to find the optimal regularization strength

## Running Experiments (Task 3)

### Basic Tests

To run basic tests for the LogRegCCD algorithm:

```bash
python some_tests.py
```

This will train a model on synthetic data and evaluate it using the measure you select when prompted.

### Testing Correctness (Lambda=0)

To verify the correctness of the implementation by testing with lambda=0 (no regularization):

```bash
python test_lambda_0.py
```

This script runs LogRegCCD with lambda=0 on all datasets and saves the results to `lambda_zero_performance.csv`.

### Comparing LogRegCCD with Standard LogisticRegression (Task 3.2)

To compare the performance of LogRegCCD and standard LogisticRegression on real datasets:

```bash
python task_3_2.py
```

This script will:
- Train both models on real datasets
- Compare their performance using ROC AUC, PR AUC, F1-score, and Balanced Accuracy
- Compare coefficient values between the two models
- Save plots to the `plots` directory

### Testing LogRegCCD with L1 Penalty Against Scikit-Learn's Implementation

To compare LogRegCCD with scikit-learn's LogisticRegression with L1 penalty:

```bash
python test_vs_scikit_with_l1.py
```

This script evaluates both implementations on real datasets and compares their performance and coefficients.

### Analyzing Effects of Synthetic Data Parameters (Task 3.1)

To analyze how parameters n, p, d, and g affect model performance:

```bash
python task_3_1.py
```

This script:
- Varies each parameter (n=number of observations, p=class prior probability, d=feature dimensionality, g=covariance parameter)
- Evaluates both LogRegCCD and LogisticRegression models
- Generates plots showing how each parameter affects model performance
- Saves results to the `plots` directory

## Output

All scripts save results and plots to:
- `plots/`: Contains all generated plots
- `lambda_zero_performance.csv`: Performance metrics for lambda=0 tests
