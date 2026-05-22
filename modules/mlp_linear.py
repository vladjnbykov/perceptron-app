import numpy as np


class MLPLinear:
    def __init__(self, hidden_neurons=4, learning_rate=0.1, random_state=42):
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.random_state = random_state

    def initialize(self, n_features):
        rng = np.random.default_rng(self.random_state)

        self.W1 = rng.normal(0, 0.5, size=(n_features, self.hidden_neurons))
        self.b1 = np.zeros((1, self.hidden_neurons))

        self.W2 = rng.normal(0, 0.5, size=(self.hidden_neurons, 1))
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        # Ingen aktiveringsfunktion i hidden layer
        h = X @ self.W1 + self.b1

        # Output layer
        z_out = h @ self.W2 + self.b2
        y_hat = self.sigmoid(z_out)

        return h, z_out, y_hat

    def predict_proba(self, X):
        _, _, y_hat = self.forward(X)
        return y_hat.ravel()

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
            h, z_out, y_hat = self.forward(X)

            loss = -np.mean(
                y * np.log(y_hat + 1e-8) +
                (1 - y) * np.log(1 - y_hat + 1e-8)
            )

            dz_out = y_hat - y

            dW2 = h.T @ dz_out / n
            db2 = np.mean(dz_out, axis=0, keepdims=True)

            dh = dz_out @ self.W2.T

            dW1 = X.T @ dh / n
            db1 = np.mean(dh, axis=0, keepdims=True)

            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1

            predictions = self.predict(X)
            accuracy = np.mean(predictions == y.ravel())

            history["loss"].append(loss)
            history["accuracy"].append(accuracy)
            history["params"].append({
                "W1": self.W1.copy(),
                "b1": self.b1.copy(),
                "W2": self.W2.copy(),
                "b2": self.b2.copy(),
            })

        return history

    def set_params(self, params):
        self.W1 = params["W1"].copy()
        self.b1 = params["b1"].copy()
        self.W2 = params["W2"].copy()
        self.b2 = params["b2"].copy()