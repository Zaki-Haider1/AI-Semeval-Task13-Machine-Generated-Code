from sklearn.naive_bayes import MultinomialNB

class NBModel:
    def __init__(self):
        self.model = MultinomialNB()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
