from sklearn.linear_model import LogisticRegression

class LogRegModel:
    
    def __init__(self, max_iter=2000, n_jobs=-1):
        self.model = LogisticRegression(max_iter=max_iter, n_jobs=n_jobs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
