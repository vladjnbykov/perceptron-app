import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(model, X, y, title="Decision boundary"):
    x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
    y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.contourf(xx, yy, preds, alpha=0.25)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    return fig


def plot_training_history(history, model_name):
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(history["accuracy"], label="Accuracy")

    if "loss" in history:
        ax.plot(history["loss"], label="Loss")

    ax.set_title(f"Training history: {model_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.legend()

    return fig