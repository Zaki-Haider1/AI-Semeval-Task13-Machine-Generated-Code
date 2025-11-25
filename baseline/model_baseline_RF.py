from sklearn.ensemble import RandomForestClassifier

class RFModel:
    
    def __init__(self, n_estimators=100, random_state=42, n_jobs=-1):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
