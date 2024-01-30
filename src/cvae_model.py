import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import cv2
from log_setup import logger

# Tensorflow Imports
import tensorflow as tf
from tensorflow.keras.layers import Activation, MaxPooling2D
from tensorflow.keras.layers import SeparableConv2D, Conv2DTranspose
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras.utils import Sequence

disable_eager_execution()
print(f"TensorFlow version: {tf.__version__}")


class CVAEDataGenerator(Sequence):
    """
    A data generator for the Conditional Variational Autoencoder (CVAE) model.

    This class generates batches of images and corresponding labels from a given set of data paths and labels.
    It shuffles the data at the end of each epoch to ensure that the model sees all data in each epoch.

    Attributes:
        data_paths: A list of paths to the data files.
        labels: A list of corresponding labels for the data files.
        batch_size: The number of samples per gradient update.
        input_shape: The shape of the input data.
        num_frames: The total number of frames in the data.

    Methods:
        __init__: Initializes the data generator.
        __len__: Returns the number of batches in the data.
        __getitem__: Returns a batch of images and labels.
        on_epoch_end: Shuffles the data at the end of each epoch.
        load_and_preprocess_data: Loads and preprocesses a batch of images and labels.
        load_preprocess_mask: Loads and preprocesses a single mask image.
    """

    def __init__(self, data_paths, labels, batch_size, input_shape, last_frame):
        """
        Initializes the data generator.

        Args:
            data_paths: A list of paths to the data files.
            labels: A list of corresponding labels for the data files.
            batch_size: The number of samples per gradient update.
            input_shape: The shape of the input data.
            last_frame: The total number of frames in the data.
        """
        self.data_paths = data_paths
        self.labels = labels
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_frames = last_frame
        self.on_epoch_end()

    def __len__(self):
        """
        Returns the number of batches in the data.

        Returns:
            The number of batches in the data.
        """
        return int(np.ceil(len(self.data_paths) / self.batch_size))

    def __getitem__(self, index):
        """
        Returns a batch of images and labels.

        Args:
            index: The index of the batch.

        Returns:
            A batch of images and labels.
        """
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_data_paths = self.data_paths[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]

        batch_images, batch_labels = self.load_and_preprocess_data(
            batch_data_paths, batch_labels
        )
        return [batch_images, batch_labels], batch_images

    def on_epoch_end(self):
        """
        Shuffles the data at the end of each epoch.
        """
        indices = np.arange(len(self.data_paths))
        np.random.shuffle(indices)
        self.data_paths = [self.data_paths[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

    def load_and_preprocess_data(self, batch_data_paths, batch_labels):
        """
        Loads and preprocesses a batch of images and labels.

        Args:
            batch_data_paths: A list of paths to the data files.
            batch_labels: A list of corresponding labels for the data files.

        Returns:
            A batch of images and labels.
        """
        batch_images = []
        batch_labels_processed = []
        for data_path, label in zip(batch_data_paths, batch_labels):
            image, label = self.load_preprocess_mask(
                data_path, label, self.input_shape, self.num_frames
            )
            batch_images.append(image)
            batch_labels_processed.append(label)
        return np.array(batch_images), np.array(batch_labels_processed)

    def load_preprocess_mask(self, mask_path, label, output_dims, last_frame):
        """
        Loads and preprocesses a single mask image.

        Args:
            mask_path: The path to the mask file.
            label: The corresponding label for the mask file.
            output_dims: The desired dimensions of the mask.
            last_frame: The total number of frames in the data.

        Returns:
            A preprocessed mask image and its corresponding label.
        """
        # Check if the file exists
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"No such file: '{mask_path}'")

        # Read and decode the image
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize the image
        mask = cv2.resize(mask, output_dims)

        # Add channel dimension
        mask = np.expand_dims(mask, axis=-1)

        # Normalize the mask
        mask = (mask / 127.5) - 1

        # Normalize the label
        label = label / last_frame
        label = np.expand_dims(label, axis=-1)

        return mask, label


class CVAEComponents:
    def __init__(self):
        self.z_mean = None
        self.z_log_var = None

    def deconv_block(self, input, filters, f_init="he_normal"):
        """
        Apply two convolutional layers with ReLU activation function.

        Args:
            input (tensor): Input tensor to the block.
            filters (int): Number of filters in the convolutional layers.

        Returns:
            tensor: Output tensor of the block with ReLU activation.
        """
        x = Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=2,
            kernel_initializer=f_init,
            data_format="channels_last",
            padding="same",
        )(input)
        x = Activation(tf.nn.leaky_relu)(x)

        x = SeparableConv2D(
            filters,
            kernel_size=(4, 4),
            depthwise_initializer=f_init,
            pointwise_initializer=f_init,
            padding="same",
        )(x)
        x = Activation(tf.nn.leaky_relu)(x)

        x = SeparableConv2D(
            filters,
            kernel_size=(4, 4),
            depthwise_initializer=f_init,
            pointwise_initializer=f_init,
            padding="same",
        )(x)
        activation = Activation(tf.nn.leaky_relu)(x)

        return activation

    def conv_block(self, input, filters, f_init="he_normal"):
        """
        Apply two convolutional layers with ReLU activation function.

        Args:
            input (tensor): Input tensor to the block.
            filters (int): Number of filters in the convolutional layers.

        Returns:
            tensor: Output tensor of the block with ReLU activation.
        """
        x = SeparableConv2D(
            filters,
            kernel_size=(4, 4),
            depthwise_initializer=f_init,
            pointwise_initializer=f_init,
            padding="same",
        )(input)
        x = Activation(tf.nn.leaky_relu)(x)

        x = SeparableConv2D(
            filters,
            kernel_size=(4, 4),
            depthwise_initializer=f_init,
            pointwise_initializer=f_init,
            padding="same",
        )(x)
        ativ = Activation(tf.nn.leaky_relu)(x)

        m_pool = MaxPooling2D(
            pool_size=(2, 2), strides=2, data_format="channels_last", padding="same"
        )(ativ)

        return m_pool

    def sampler(self, args):
        """
        Reparameterization trick by sampling from an isotropic unit Gaussian.

        Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        Returns:
            z (tensor): sampled latent vector
        """
        self.z_mean, self.z_log_var = args
        batch = K.shape(self.z_mean)[0]
        dim = K.int_shape(self.z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return self.z_mean + K.exp(0.5 * self.z_log_var) * epsilon

    def mse_kl_loss(self, y_true, y_pred, beta: float = 1.0):
        """Calculate loss = reconstruction loss + KL loss for each data in minibatch"""
        # E[log P(X|z)]
        squared_difference = tf.square(y_true - y_pred)
        reconstruction = tf.reduce_mean(squared_difference, axis=-1)
        # D_KL(Q(z|X) || P(z|X)); calculate in closed from as both dist. are Gaussian
        kl_divergence = 0.5 * tf.reduce_sum(
            tf.exp(self.z_log_var) + tf.square(self.z_mean) - 1.0 - self.z_log_var,
            axis=-1,
        )
        return reconstruction + beta * kl_divergence


class ReduceLROnPlateauSteps(tf.keras.callbacks.ReduceLROnPlateau):
    def __init__(
        self, monitor="val_loss", factor=0.5, patience=500, min_lr=1e-8, **kwargs
    ):
        super().__init__(
            monitor=monitor, factor=factor, patience=patience, min_lr=min_lr, **kwargs
        )
        self.wait = 0
        self.best = 0

    def on_train_batch_end(self, batch, logs=None):
        current = self.get_monitor_value(logs)
        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.wait = 0
                new_lr = self.model.optimizer.learning_rate * self.factor
                new_lr = tf.keras.backend.get_value(new_lr)
                self.model.optimizer.learning_rate = new_lr
                print("Reducing learning rate to %s." % (new_lr,))

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logger.warning(
                "Learning rate reduction on plateau conditioned on metric `%s` "
                "which is not available. Available metrics are: %s"
                % (self.monitor, ", ".join(list(logs.keys()))),
                RuntimeWarning,
            )
        return monitor_value


class EarlyStoppingSteps(tf.keras.callbacks.EarlyStopping):
    def __init__(self, monitor="val_loss", min_delta=0, patience=500, **kwargs):
        super().__init__(
            monitor=monitor, min_delta=min_delta, patience=patience, **kwargs
        )
        self.wait = 0
        self.best = 0

    def on_train_batch_end(self, batch, logs=None):
        """
        At the end of each batch, check if the monitored quantity has improved.
        If number of batches since the last improvement is more than the patience,
        stop training.
        """
        current = self.get_monitor_value(logs)
        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.wait = 0
                self.stopped_epoch = self.model.history.epoch[-1]
                self.model.stop_training = True
                print("Early stopping")

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logger.warning(
                "Early stopping conditioned on metric `%s` "
                "which is not available. Available metrics are: %s"
                % (self.monitor, ", ".join(list(logs.keys()))),
                RuntimeWarning,
            )
        return monitor_value


class ModelCheckpointSteps(tf.keras.callbacks.ModelCheckpoint):
    def __init__(
        self,
        filepath,
        monitor="val_loss",
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        verbose=0,
        save_freq="epoch",
        **kwargs,
    ):
        super().__init__(
            filepath=filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            mode=mode,
            verbose=verbose,
            save_freq=save_freq,
            **kwargs,
        )
        self.step_count = 0

    def on_train_batch_end(self, batch, logs=None):
        self.step_count += 1
        if self.step_count % self.save_freq == 0:
            self.save_weights(self.filepath, overwrite=True)


class HistoryLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_interval):
        super().__init__()
        self.log_interval = log_interval
        self.step_count = 0
        self.history = []

    def on_train_batch_end(self, batch, logs=None):
        self.step_count += 1
        if self.step_count % self.log_interval == 0:
            self.history.append(logs)

    def get_history(self):
        return self.history
