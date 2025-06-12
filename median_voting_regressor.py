import numpy as np

class MedianVotingRegressor:
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return np.median(predictions, axis=0)
