from sklearn.svm import SVC

class SVMClassifier:
    def __init__(self, kernel='linear', C=1.0, max_iter=-1):
        self.model = SVC(kernel=kernel, C=C, max_iter=max_iter)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
