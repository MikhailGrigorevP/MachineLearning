# Обучение персептрона
# 1. Инициализовать веса нулями или небольшими случайными числами
# 2. Для x(i) найти y^, обновить веса
# Wj = Wj + dWj
# dWj = n(y(i)-y^(i)) * x(i)j, n = (0, 1)
# Классы линейно сепарабельны

import numpy as np


class Perceptron(object):

    """
    Parameters
    ---------
    eta: float  - learning rate
    n_iter: int  - step
    random_state: int  - initial random state

    Attributes
    ---------
    w_: np.array(1)  - weights
    /// where w_[0] - bias (vector b)
    errors: list  - wrong classifications
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

        # Fit
        self.w_ = None
        self.errors_ = None

    def fit(self, X, y):

        """
        Parameters
        ---------
        X: {array-like}, shape = [n_samples - образцы, n_features - признаки] - обучающие векторы
        y: {array- like}, shape = [n_samples] - целевые значения

        Returns
        ---------
        self: object
        """

        random_generator = np.random.RandomState(self.random_state)
        self.w_ = random_generator.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0)
            self.errors_.append(errors)

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
