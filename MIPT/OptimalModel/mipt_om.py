import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
columns = ('age workclass fnlwgt education educ-num marital-status occupation relationship '
           'race sex capital-gain capital-loss  hours-per-week native-country salary')
numeric_indices = np.array([0, 2, 4, 10, 11, 12])
categorical_indices = np.array([1, 3, 5, 6, 7, 8, 9, 13])
df.columns = columns.split()
df = df.replace('?', np.nan)
df = df.dropna()
df['salary'] = df['salary'].apply(
    (lambda x: x == ' >50K'))  # Будем предсказывать 1 (True), если зарплата больше 50K, 0 (False) иначе

numeric_data = df[df.columns[numeric_indices]]
categorial_data = df[df.columns[categorical_indices]]
print(categorial_data.head())

dummy_features = pd.get_dummies(categorial_data)
X = pd.concat([numeric_data, dummy_features], axis=1)
print(X.head())

y = df['salary']
print(y.mean())

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, train_size=0.8)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def plot_roc_curve(model, X_train, X_test, y_train, y_test):
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    plt.figure(figsize=(12, 10))

    print(f'Train roc-auc: {roc_auc_score(y_train, y_train_proba)}')
    print(f'Test roc-auc: {roc_auc_score(y_test, y_test_proba)}')

    plt.plot(*roc_curve(y_train, y_train_proba)[:2], label='train roc-curve')
    plt.plot(*roc_curve(y_test, y_test_proba)[:2], label='test roc-curve')

    plt.plot([0, 1], [0, 1], linestyle='--', color='black')
    plt.grid(True)
    plt.legend()
    plt.show()


model = LogisticRegression().fit(X_train, y_train)
plot_roc_curve(model, X_train, X_test, y_train, y_test)


def search(X, y, model, param_name, grid, draw=True):
    parameters = {param_name: grid}

    CV_model = GridSearchCV(estimator=model, param_grid=parameters, cv=5, scoring='roc_auc', n_jobs=-1)
    CV_model.fit(X, y)
    means = CV_model.cv_results_['mean_test_score']
    error = CV_model.cv_results_['std_test_score']

    if draw:
        plt.figure(figsize=(15, 8))
        plt.title('choose ' + param_name)

        plt.plot(grid, means, label='mean values of score')

        plt.fill_between(grid, means - 2 * error, means + 2 * error, color='green',
                         label='deviation area between errors')
        plt.legend()
        plt.xlabel('parameter')
        plt.ylabel('roc_auc')
        plt.show()

    return means, error


models = [KNeighborsClassifier(n_jobs=-1)]
param_names = ['n_neighbors']
grids = [np.array(np.linspace(4, 25, 10), dtype='int')]
param_scales = ['ordinary']

for model, param_name, grid, param_scale in zip(models,
                                                param_names,
                                                grids,
                                                param_scales):
    search(X_train, y_train, model, param_name, grid, param_scale)

model = KNeighborsClassifier(n_neighbors=25, n_jobs=-1).fit(X_train, y_train)
plot_roc_curve(model, X_train, X_test, y_train, y_test)

models = [DecisionTreeClassifier()]
param_names = ['max_depth']
grids = [np.arange(3, 21, 2)]
param_scales = ['ordinary']

for model, param_name, grid, param_scale in zip(models,
                                                param_names,
                                                grids,
                                                param_scales):
    search(X_train, y_train, model, param_name, grid, param_scale)

model = DecisionTreeClassifier(max_depth=7).fit(X_train, y_train)
plot_roc_curve(model, X_train, X_test, y_train, y_test)

models = [RandomForestClassifier(n_jobs=-1)]
param_names = ['n_estimators']
grids = [[10, 20, 30, 50, 75, 100, 150]]
param_scales = ['ordinary']

for model, param_name, grid, param_scale in zip(models,
                                                param_names,
                                                grids,
                                                param_scales):
    search(X_train, y_train, model, param_name, grid, param_scale)

model = RandomForestClassifier(n_estimators=100, n_jobs=-1).fit(X_train, y_train)
plot_roc_curve(model, X_train, X_test, y_train, y_test)

models = [xgboost.XGBClassifier(n_estimators=200)]
param_names = ['max_depth']
grids = [np.arange(3, 10, 2)]
param_scales = ['ordinary']

for model, param_name, grid, param_scale in zip(models,
                                                param_names,
                                                grids,
                                                param_scales):
    search(X_train, y_train, model, param_name, grid, param_scale)

model = xgboost.XGBClassifier(max_depth=5, n_estimators=300).fit(X_train, y_train)
plot_roc_curve(model, X_train, X_test, y_train, y_test)