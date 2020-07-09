import numpy as np


class AdalineSGD(object):

    """
    Parameters
    ---------
    eta: float  - learning rate
    n_iter: int  - step
    random_state: int  - initial random state
    shuffle: bool - shuffle data

    Attributes
    ---------
    w_: np.array(1)  - weights
    /// where w_[0] - bias (vector b)
    cost_: list  -
    """

    def __init__(self, eta=0.01, n_iter=10,
                 shuffle=True, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
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

        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

    def _shuffle(self, X, y):
        r = self.random_generator.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.random_generator = np.random.RandomState(self.random_state)
        self.w_ = self.random_generator.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
