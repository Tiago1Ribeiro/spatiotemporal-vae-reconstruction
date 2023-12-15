"""
This module contains functions for visualizing data and models.
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(history):
    """
    Plots learning curves for a trained model.

    This function takes a History object as input and generates loss and accuracy
    plots for the training and validation sets.

    Parameters:
    history (History): The history object obtained from the fit method of a keras Model instance.

    Raises:
    ValueError: If the history object does not contain 'loss' or 'accuracy' keys.

    Example:
    >>> history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
    >>> plot_learning_curves(history)
    """
    setup_plot()
    plot_loss(history)
    plot_min_loss_points(history)
    plot_learning_rate_changes(history)
    plt.show()


def setup_plot():
    plt.figure(figsize=(15, 6))
    plt.title("Learning Curves")
    plt.grid(True, linestyle="-.", linewidth=0.3)
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")


def plot_loss(history):
    plt.xticks(np.arange(0, max(history.epoch) + 50, 50))
    plt.yticks(np.arange(0, max(history.history["loss"]), 0.1), fontsize=8)
    plt.xticks(fontsize=8)
    plt.plot(history.history["loss"], label="train loss", linewidth=0.7)
    plt.plot(history.history["val_loss"], label="test loss", linewidth=0.7)
    plt.legend()


def plot_min_loss_points(history):
    min_loss, min_loss_epoch = min_and_index(history.history["loss"])
    min_val_loss, min_val_loss_epoch = min_and_index(history.history["val_loss"])
    plot_min_loss_point(min_loss, min_loss_epoch, "red")
    plot_min_loss_point(min_val_loss, min_val_loss_epoch, "green")


def min_and_index(lst):
    min_val = min(lst)
    min_index = lst.index(min_val)
    return min_val, min_index


def plot_min_loss_point(min_loss, min_loss_epoch, color):
    plt.plot(min_loss_epoch, min_loss, marker="o", markersize=2, color=color)
    plt.annotate(
        f"min: {min_loss:.2e}",
        xy=(min_loss_epoch, min_loss),
        xytext=(min_loss_epoch + 10, min_loss + 0.01),
        fontsize=7,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
    )

def plot_learning_rate_changes(history):
    if "lr" in history.history.keys():
        lr_changes = find_learning_rate_changes(history)
        for i in lr_changes:
            plt.axvline(x=i, color="mediumseagreen", linestyle="--", linewidth=0.7)
            plt.text(
                i + 5,
                min(history.history["loss"]) + 0.03,
                f"l.r.: {history.history['lr'][i]:.0e}",
                fontsize=7,
                color="black",
                rotation=90,
            )

def find_learning_rate_changes(history):
    return [
        i
        for i in range(len(history.history["lr"]))
        if history.history["lr"][i] != history.history["lr"][i - 1]
    ]
