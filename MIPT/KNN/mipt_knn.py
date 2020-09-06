import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_min, x_max = -1, 1
y_min, y_max = -1, 1
h = 0.05
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])


def devil(N, D=2, K=3):
    """
    :param N: points
    :param D: dimensions
    :param K: classes
    :return:
    """
    N = 100
    D = 2
    K = 3
    X = np.zeros((N * K, D))
    y = np.zeros(N * K, dtype='uint8')

    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X, y


def draw(a, b, y):

    plt.figure(figsize=(12, 12))
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    plt.scatter(a, b, c=y)
    plt.grid(True)
    plt.show()


def draw_2(model, X_test, y_test):

    plt.figure(figsize=(12, 12))
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, label='Тестовые точки')

    plt.legend()

    plt.show()


def main():
    X, y = devil(300)

    # Noise
    X[:, 0] += np.random.normal(loc=0, scale=0.15, size=300)
    X[:, 1] += np.random.normal(loc=0, scale=0.15, size=300)

    # graphics
    # draw(X[:, 0], X[:, 1], y)

    # splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)

    # model
    model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f'Процент правильно угаданных ответов на обучающем множестве: {accuracy_score(y_train, y_train_pred)}')
    print(f'Процент правильно угаданных ответов на тестовом множестве: {accuracy_score(y_test, y_test_pred)}')

    draw_2(model, X_test, y_test)


if __name__ == "__main__":
    main()
