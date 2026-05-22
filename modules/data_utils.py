import numpy as np
from sklearn.datasets import make_moons, make_circles


def make_linear_data(n_samples=100, noise=0.2, random_state=42):
    rng = np.random.default_rng(random_state)

    X_pos = rng.normal(loc=[1.5, 1.5], scale=noise, size=(n_samples // 2, 2))
    X_neg = rng.normal(loc=[-1.5, -1.5], scale=noise, size=(n_samples // 2, 2))

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(n_samples // 2), np.zeros(n_samples // 2)])

    return X, y


def make_xor_data(n_samples=200, noise=0.15, random_state=42):
    rng = np.random.default_rng(random_state)

    X = rng.uniform(-2, 2, size=(n_samples, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)

    X += rng.normal(0, noise, size=X.shape)

    return X, y


def make_moons_data(n_samples=200, noise=0.2, random_state=42):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y


def make_circles_data(n_samples=200, noise=0.1, random_state=42):
    X, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=0.5,
        random_state=random_state,
    )
    return X, y


def get_dataset(name, n_samples=200, noise=0.2, random_state=42):
    if name == "Linear":
        return make_linear_data(n_samples, noise, random_state)
    if name == "XOR":
        return make_xor_data(n_samples, noise, random_state)
    if name == "Moons":
        return make_moons_data(n_samples, noise, random_state)
    if name == "Circles":
        return make_circles_data(n_samples, noise, random_state)

    raise ValueError(f"Unknown dataset: {name}")