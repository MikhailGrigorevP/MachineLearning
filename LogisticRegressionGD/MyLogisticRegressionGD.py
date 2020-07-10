import numpy as np


class LogisticRegressionGD(object):

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
    cost_: list  -
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

        # Fit
        self.w_ = None
        self.cost_ = None

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
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (-y.dot(np.log(output)) -
                    ((1-y).dot(np.log(1 - output))))
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
