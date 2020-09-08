import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

y_true = np.array([0,1,1,0,0,1,1,0,0,0])
y_scores = np.array([0.35, 0.85, 0.75, 0.25, 0.05, 0.45, 0.95, 0.65, 0.15, 0.55])
print(roc_auc_score(y_true, y_scores))

y_true = [10, 2, -1, 3, 7]
y_pred = [8, 1, 3, -1, 6]
print(mean_squared_error(y_true, y_pred, squared=True))
