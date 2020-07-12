from sklearn import datasets
from sklearn.model_selection import train_test_split
from LogisticRegressionGD.MyLogisticRegressionGD import LogisticRegressionGD
from Algorythms.plotDecisionRegions import plot_decision_regions
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

X_train_01_subset = X_train[(y_train == 0) | y_train == 1]
y_train_01_subset = y_train[(y_train == 0) | y_train == 1]
lrgd = LogisticRegressionGD(eta=0.05,
                            n_iter=1000,
                            random_state=1)
lrgd.fit(X_train_01_subset,
         y_train_01_subset)
plot_decision_regions(X=X_train_01_subset,
                      y=y_train_01_subset,
                      classifier=lrgd)
plt.xlabel('Длина лепестка [std]')
plt.ylabel('Ширина лепестка [std]')
plt.legend(loc='upper left')
plt.show()
