from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from Algorithms.plotDecisionRegions import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

print("Метки классов:", np.unique(y))

# Разбили набор данных на обучающий и испытательный

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Проверить количество меток

print("Метки y:", np.bincount(y))
print("Метки y_train:", np.bincount(y_train))
print("Метки y_test:", np.bincount(y_test))

# Масштабирование признаков

print("// Обучение начато //")
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
print("// Обучение завершено //")

# Логистическая регрессия

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier=lr,
                      test_idx=range(105, 150))
plt.xlabel('Длина лепестка [std]')
plt.ylabel('Ширина лепестка [std]')
plt.legend(loc='upper left')
plt.show()

# Вероятности

lr.predict_proba(X_test_std[:3, :])

# Настройка переобучения

# weights, params = [], []
# for c in np.arange(-5, 5):
#     lr = LogisticRegression(C=10.**c, random_state=1)
#     lr.fit(X_train_std, y_train)
#     weights.append(lr.coef_[1])
#     params.append(10.**c)
# weights = np.array(weights)
# plt.plot(params, weights[:, 0],
#          label='Длина лепестка')
# plt.plot(params, weights[:, 1], linestyle='--',
#          label='Ширина лепестка')
# plt.ylabel('Весовой коэффициент')
# plt.xlabel('C')
# plt.legend(loc='upper left')
# plt.xscale('log')
# plt.show()

