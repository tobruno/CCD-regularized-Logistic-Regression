import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, make_classification
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def data_preprocessing(df):

    X = df.drop(columns=["target"])
    y = df.target
    #print(X.info())

    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    if len(cat_features) != 0:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features)
        ])
    else: 
        preprocessor = ColumnTransformer([
            ("num", num_pipeline, num_features)
        ])


    # Fit & transform training data, transform test data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Convert back to DataFrame
    X_train_final = pd.DataFrame(X_train_processed, columns=preprocessor.get_feature_names_out())
    X_test_final = pd.DataFrame(X_test_processed, columns=preprocessor.get_feature_names_out())

    return X_train_final, X_test_final, y_train, y_test



def get_cancer_data():
    data_cancer = load_breast_cancer()
    breast_cancer = pd.DataFrame(data_cancer.data, columns=data_cancer.feature_names)
    breast_cancer['target'] = data_cancer.target

    X_train, X_test, y_train, y_test = data_preprocessing(breast_cancer)
    return X_train, X_test, y_train, y_test

def get_wine_data():
    data_wine = load_wine()
    wine = pd.DataFrame(data_wine.data, columns=data_wine.feature_names)
    wine['target'] = data_wine.target
    wine = wine[wine.target != 2]

    X_train, X_test, y_train, y_test = data_preprocessing(wine)
    # values, counts = np.unique(y_test, return_counts=True)
    # print(values, counts)
    return X_train, X_test, y_train, y_test

def get_heart_data():
    data = fetch_ucirepo(id=45) 
    heart_data = data.data.features 
    target = data.data.targets
    heart_data['target'] = target
    heart_data.target = heart_data.target.replace({2: 1, 3: 1, 4: 1})

    print(np.unique(heart_data.target, return_counts=True))

    X_train, X_test, y_train, y_test = data_preprocessing(heart_data)
    return X_train, X_test, y_train, y_test

def get_titanic_data():
    df = sns.load_dataset("titanic")
    titanic = df.rename(columns={'survived':'target'})
    titanic = titanic.drop(columns=["deck", "fare"])

    X_train, X_test, y_train, y_test = data_preprocessing(titanic)
    return X_train, X_test, y_train, y_test


def get_synthetic_data():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, n_classes=2, random_state=42)
    column_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=column_names)
    df["target"] = y

    X_train, X_test, y_train, y_test = data_preprocessing(df)
    return X_train, X_test, y_train, y_test


# QUICK TESTS

# print('Cancer')
# X_train, X_test, y_train, y_test = get_cancer_data()
# lg = LogisticRegression()
# lg.fit(X_train, y_train)

# pred = lg.predict(X_test)
# print(accuracy_score(y_test, pred))

print('Heart')
X_train, X_test, y_train, y_test = get_heart_data()
lg = LogisticRegression()
lg.fit(X_train, y_train)

pred = lg.predict(X_test)
print(accuracy_score(y_test, pred))


# print('Wine')
# X_train, X_test, y_train, y_test = get_wine_data()
# lg = LogisticRegression()
# lg.fit(X_train, y_train)

# pred = lg.predict(X_test)
# print(accuracy_score(y_test, pred))


# print('Titanic')
# X_train, X_test, y_train, y_test = get_titatnic_data()
# lg = LogisticRegression()
# lg.fit(X_train, y_train)

# pred = lg.predict(X_test)
# print(accuracy_score(y_test, pred))

# print('Synthetic')
# X_train, X_test, y_train, y_test = get_synthetic_data()
# lg = LogisticRegression()
# lg.fit(X_train, y_train)

# pred = lg.predict(X_test)
# print(accuracy_score(y_test, pred))






