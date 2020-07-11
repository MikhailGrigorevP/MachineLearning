from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from plotDecisionRegions import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# Разбили набор данных на обучающий и испытательный

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

# Tree classifier

knn = KNeighborsClassifier(n_neighbors=5,
                           p=2,
                           metric='minkowski')
knn.fit(X_train, y_train)
plot_decision_regions(X_combined,
                      y_combined,
                      classifier=knn,
                      test_idx=range(105, 150))
plt.xlabel('Длина лепестка [std]')
plt.ylabel('Ширина лепестка [std]')
plt.legend(loc='upper left')
plt.show()
