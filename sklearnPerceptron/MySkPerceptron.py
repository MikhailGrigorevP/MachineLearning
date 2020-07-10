from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from plotDecisionRegions import plot_decision_regions
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

# Перцептрон

ppn = Perceptron(max_iter=40, eta0=0.01, random_state=1)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)

print("Неправильно классифицированные: %d" % (y_test != y_pred).sum())

# Правильность

print("Правильность: %.2f" % accuracy_score(y_test, y_pred))


# График

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))
plt.xlabel('Длина лепестка [std]')
plt.ylabel('Ширина лепестка [std]')
plt.legend(loc='upper left')
plt.show()

