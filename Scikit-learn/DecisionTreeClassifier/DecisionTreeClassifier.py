from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
from sklearn.tree import DecisionTreeClassifier
from Algorythms.plotDecisionRegions import plot_decision_regions
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

tree = DecisionTreeClassifier(criterion='gini',
                              max_depth=4,
                              random_state=1)
tree.fit(X_train, y_train)
plot_decision_regions(X_combined,
                      y_combined,
                      classifier=tree,
                      test_idx=range(105, 150))
plt.xlabel('Длина лепестка [std]')
plt.ylabel('Ширина лепестка [std]')
plt.legend(loc='upper left')
plt.show()

dot_data = export_graphviz(tree, filled=True,
                           rounded=True,
                           class_names=['Setosa',
                                        'Versicolor',
                                        'Virginica'],
                           feature_names=['petal length',
                                          'petal width'],
                           out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')


