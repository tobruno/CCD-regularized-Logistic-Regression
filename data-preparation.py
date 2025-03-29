import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, make_classification
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo 

def data_preprocessing(df):
    target = df.target
    df = df.drop(columns=["target"])
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    df[numerical_columns] = imputer.fit_transform(df[numerical_columns])
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


    categorical_columns = df.select_dtypes(include=[object, 'category']).columns
    if len(categorical_columns) != 0:
        imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_columns] = imputer.fit_transform(df[categorical_columns])
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    df['target'] = target

    return df

def get_cancer_data():
    data_cancer = load_breast_cancer()
    breast_cancer = pd.DataFrame(data_cancer.data, columns=data_cancer.feature_names)
    breast_cancer['target'] = data_cancer.target

    breast_cancer = data_preprocessing(breast_cancer)   
    return breast_cancer

def get_wine_data():
    data_wine = load_wine()
    wine = pd.DataFrame(data_wine.data, columns=data_wine.feature_names)
    wine['target'] = data_wine.target

    wine = data_preprocessing(wine)
    return wine

def get_heart_data():
    data = fetch_ucirepo(id=45) 
    heart_data = data.data.features 
    target = data.data.targets 
    heart_data['target'] = target

    print(heart_data.target.value_counts())

    heart_data = data_preprocessing(heart_data)   
    return heart_data

def get_titatnic_data():
    df = sns.load_dataset("titanic")
    titanic = df.rename(columns={'survived':'target'})
    titanic = titanic.drop(columns=["deck"])

    titanic = data_preprocessing(titanic)
    return titanic


def get_synthetic_data():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, n_classes=2, random_state=42)
    column_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=column_names)
    df["target"] = y
    return df


# QUICK TESTS

# df = get_cancer_data()
# df.info()
# X = df.drop(columns = ['target'])
# y = df.target
# lg = LogisticRegression()
# lg.fit(X, y)

# pred = lg.predict(X)
# print(accuracy_score(y, pred))

# df = get_heart_data()
# df.info()
# X = df.drop(columns = ['target'])
# y = df.target
# lg = LogisticRegression()
# lg.fit(X, y)

# pred = lg.predict(X)
# print(accuracy_score(y, pred))

# df = get_synthetic_data()
# df.info()
# X = df.drop(columns = ['target'])
# y = df.target
# lg = LogisticRegression()
# lg.fit(X, y)

# pred = lg.predict(X)
# print(accuracy_score(y, pred))

# df = get_titatnic_data()
# df.info()
# X = df.drop(columns = ['target'])
# y = df.target
# lg = LogisticRegression()
# lg.fit(X, y)

# pred = lg.predict(X)
# print(accuracy_score(y, pred))

# df = get_wine_data()
# df.info()
# X = df.drop(columns = ['target'])
# y = df.target
# lg = LogisticRegression()
# lg.fit(X, y)

# pred = lg.predict(X)
# print(accuracy_score(y, pred))






