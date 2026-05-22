import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, random_state=42):
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.weights = None
        self.bias = None

    def initialize(self, n_features):
        rng = np.random.default_rng(self.random_state)
        self.weights = rng.normal(0, 0.1, size=n_features)
        self.bias = 0.0

    def activation(self, z):
        return (z >= 0).astype(int)

    def predict(self, X):
        z = X @ self.weights + self.bias
        return self.activation(z)

    def fit(self, X, y, epochs=50):
        self.initialize(X.shape[1])

        history = {
            "accuracy": [],
            "errors": [],
            "weights": [],
            "bias": [],
        }

        for _ in range(epochs):
            errors = 0

            for xi, target in zip(X, y):
                prediction = self.predict(xi.reshape(1, -1))[0]
                update = self.learning_rate * (target - prediction)

                self.weights += update * xi
                self.bias += update

                if update != 0:
                    errors += 1

            predictions = self.predict(X)
            accuracy = np.mean(predictions == y)

            history["accuracy"].append(accuracy)
            history["errors"].append(errors)
            history["weights"].append(self.weights.copy())
            history["bias"].append(self.bias)
        return history
    
    def set_params(self, weights, bias):
        self.weights = weights.copy()
        self.bias = bias     