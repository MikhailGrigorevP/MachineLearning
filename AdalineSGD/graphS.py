from AdalineSGD.MyAdalineSGD import AdalineSGD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

fig, ax = plt.subplots(nrows =1, ncols=2, figsize=(10, 4))


X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean())/ X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean())/ X[:, 1].std()

ada1 = AdalineSGD(n_iter=15, eta=0.01, random_state=1).fit(X_std, y)
ax[0].plot(range(1, len(ada1.cost_) + 1),
           ada1.cost_, marker='o')
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Cost")
ax[0].set_title('AdalineSGD - learning rate 0.01')
plt.show()