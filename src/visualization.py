"""
This module contains functions for visualizing data and models.
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.max_open_warning": 0})

# typing
from typing import List


def setup_plot(log_scale=True, plt_title: str = "Learning Curves"):
    plt.figure(figsize=(15, 6))
    plt.title(plt_title)
    plt.grid(True, linestyle="-.", linewidth=0.3)
    if log_scale:
        plt.yscale("symlog")
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


def plot_learning_curves(
    history: "keras.callbacks.History",
    log_scale: bool = True,
    plt_title: str = "Learning Curves",
):
    """
    Plots learning curves for a trained model.

    This function takes a History object as input and generates loss and accuracy
    plots for the training and validation sets.

    Parameters:
    history (History): The history object obtained from the fit method of a keras Model instance.
    log_scale (bool): Whether to use a logarithmic scale for the y-axis. Default is True.
    plt_title (str): The title of the plot. Default is "Learning Curves".

    Raises:
    ValueError: If the history object does not contain 'loss' or 'accuracy' keys.

    Example:
    >>> history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
    >>> plot_learning_curves(history)
    """
    setup_plot(log_scale, plt_title)
    plot_loss(history)
    plot_min_loss_points(history)
    plot_learning_rate_changes(history)
    plt.show()


def plot_generated_imgs(model, frames_num_list: List[int]):
    """
    Plots generated images from a trained model.

    Parameters:
    model (keras.Model or torch.nn.Module): The trained model.
    frames_num_list (List[int]): The list of frames numbers.

    Returns:
    None

    Side effects:
    Displays the generated images using matplotlib.
    """
    # Check if model and frames_num_list are not None
    if model is None:
        raise ValueError("No model was provided. Please provide a trained model.")
    if frames_num_list is None:
        raise ValueError(
            "No frames number list was provided. Please provide a list of frames numbers."
        )

    # Generate images from the latent space
    z = np.random.normal((len(frames_num_list), model.input_shape[1]))
    if hasattr(model, "forward"):
        # PyTorch model
        z = torch.from_numpy(z).float()
        generated_imgs = model.forward(z).detach().numpy()
    else:
        # Keras model
        generated_imgs = model.predict(z)

    # Scale the pixel values to the range [0, 1]
    generated_imgs = (generated_imgs + 1) / 2.0

    # Plot the generated images using the frames_num_list as reference, 2 columns
    n_rows = int(len(frames_num_list) / 2)
    _, axs = plt.subplots(n_rows, 2, figsize=(15, 15))
    axs = axs.flatten()
    for i, img in enumerate(generated_imgs):
        axs[i].imshow(img)
        axs[i].set_title(f"Frame {frames_num_list[i]}")

    plt.show()


def create_boxplot(data, positions, colors):
    """
    Creates a boxplot using matplotlib.

    Parameters:
    data (dict): A dictionary where the keys are the names of the models and the values are another dictionary containing the 'ts' and 'pr' lists.
    positions (dict): A dictionary specifying the positions for the bars.
    colors (list): A list of colors for the boxes.

    Returns:
    None
    """
    # Set up the figure and axes
    _, ax = plt.subplots(figsize=(10, 8))

    # Function to set box colors
    def set_box_colors(boxplot):
        """
        Sets the colors for the boxes in the boxplot.
        """
        for patch, color in zip(boxplot["boxes"], colors):
            patch.set_facecolor(color)

    # Create boxplots
    for key, pos in positions.items():
        boxplot = ax.boxplot(
            [data[name][key] for name in data],
            positions=pos,
            medianprops={"linewidth": 1, "color": "orange"},
            showfliers=False,
            flierprops=dict(markerfacecolor="r", markersize=2),
            patch_artist=True,
        )
        set_box_colors(boxplot)

    # Add dashed gridlines for y axis ONLY
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5)

    # x labels
    ax.set_xticks([1.5, 3.25])
    ax.set_xticklabels(["Test Set", "U-NET"])

    # Increase font size
    ax.tick_params(axis="both", which="major", labelsize=14)

    # Add legend for each model, font size 14
    ax.legend(
        [boxplot["boxes"][i] for i in range(len(data))],
        list(data.keys()),
        fontsize=14,
    )

    # Add y label
    ax.set_ylabel("Haussdorf Distance", fontsize=14)

    # Show the plot
    plt.show()
