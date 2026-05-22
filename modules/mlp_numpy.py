import numpy as np


class MLPNumPy:
    def __init__(self, hidden_neurons=4, learning_rate=0.1, random_state=42):
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def initialize(self, n_features):
        rng = np.random.default_rng(self.random_state)

        self.W1 = rng.normal(0, 1, size=(n_features, self.hidden_neurons))
        self.b1 = np.zeros((1, self.hidden_neurons))

        self.W2 = rng.normal(0, 1, size=(self.hidden_neurons, 1))
        self.b2 = np.zeros((1, 1))
        
    def set_params(self, params):
        self.W1 = params["W1"].copy()
        self.b1 = params["b1"].copy()
        self.W2 = params["W2"].copy()
        self.b2 = params["b2"].copy()    

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.sigmoid(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = self.sigmoid(z2)

        return z1, a1, z2, a2

    def predict_proba(self, X):
        _, _, _, a2 = self.forward(X)
        return a2.ravel()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def fit(self, X, y, epochs=1000):
        self.initialize(X.shape[1])
        y = y.reshape(-1, 1)

        history = {
            "loss": [],
            "accuracy": [],
            "params": [],
        }

        n = X.shape[0]

        for _ in range(epochs):
            z1, a1, z2, a2 = self.forward(X)

            loss = -np.mean(
                y * np.log(a2 + 1e-8) + (1 - y) * np.log(1 - a2 + 1e-8)
            )

            dz2 = a2 - y
            dW2 = a1.T @ dz2 / n
            db2 = np.mean(dz2, axis=0, keepdims=True)

            dz1 = (dz2 @ self.W2.T) * self.sigmoid_derivative(a1)
            dW1 = X.T @ dz1 / n
            db1 = np.mean(dz1, axis=0, keepdims=True)

            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1

            predictions = self.predict(X)
            accuracy = np.mean(predictions == y.ravel())

            history["params"].append({
                "W1": self.W1.copy(),
                "b1": self.b1.copy(),
                "W2": self.W2.copy(),
                "b2": self.b2.copy(),
            })

            history["loss"].append(loss)
            history["accuracy"].append(accuracy)

        return history