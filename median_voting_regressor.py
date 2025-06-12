import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

class MedianVotingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        self.fitted_estimators_ = []
        for name, est in self.estimators:
            est_clone = clone(est)
            est_clone.fit(X, y)
            self.fitted_estimators_.append(est_clone)
        return self

    def predict(self, X):
        preds = np.column_stack([est.predict(X) for est in self.fitted_estimators_])
        return np.median(preds, axis=1)
