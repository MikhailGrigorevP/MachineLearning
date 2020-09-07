import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

# Этот датасет описывает средние цены на недвижимость в микрорайонах Бостона в $1000
house_data = load_boston()
print(house_data['DESCR'])

# Выделим матрицу объекты-признаки в переменную  X , правильные ответы --- в переменную  y
X = pd.DataFrame(house_data['data'], columns=house_data['feature_names'])
y = house_data['target']

# Описание
X.describe()

# X.hist(X.columns, figsize=(10, 10))
# plt.figure(figsize=(10, 7))
# sns.heatmap(X.corr())

# Разобьём данные на train и test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

# Масштабируем
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Обучаем
model = LinearRegression()
model.fit(X_train, y_train)
y_train_prediction = model.predict(X_train)
y_test_prediction = model.predict(X_test)

plt.figure(figsize=(20, 8))
plt.bar(X.columns, model.coef_)

print(f'Train MSE: {mean_squared_error(y_train, y_train_prediction)}')
print(f'Test MSE: {mean_squared_error(y_test, y_test_prediction)}')

print(f'Train MAE: {mean_absolute_error(y_train, y_train_prediction)}')
print(f'Test MAE: {mean_absolute_error(y_test, y_test_prediction)}')

y.mean()

result = cross_val_score(estimator=LinearRegression(), X=X, y=y, scoring='neg_mean_absolute_error', cv=5)
print(result)
print(f'Среднее MAE равно {-result.mean()}, стандартное отклонение MAE равно {result.std()}')
