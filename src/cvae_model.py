#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple, Optional
import logging
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2DTranspose
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, Activation
from tensorflow.keras.layers import Lambda, Concatenate, Flatten
from tensorflow.keras.models import Model

logging.basicConfig(level=logging.DEBUG)


class CVAE:
    """
    Conditional Variational Autoencoder (CVAE) for image generation.

    Parameters:
        - input_shape (tuple): Shape of the input images, e.g., (height, width, channels).
        - size_label (int): Number of categorical labels for conditioning the generation.
        - train_labels (numpy.ndarray): Array of shape (num_samples, size_label)
          representing training labels.
        - config (dict): Configuration dictionary containing model hyperparameters.

    Methods:
        - __init__(self, input_shape, size_label, train_labels, config):
          Initializes the CVAE instance with the provided parameters.

        - build_cvae(self):
          Builds the CVAE model architecture using the specified hyperparameters.

        - conv_block(self, input_data, filters, f_init="he_normal"):
          Defines a convolutional block for the encoder part of the CVAE.

        - deconv_block(self, input_data, filters, f_init="he_normal"):
          Defines a deconvolutional block for the decoder part of the CVAE.

        - sampler(self, args):
          Sampling function for the latent space during training.

        - compile_cvae(self, optimizer):
          Compiles the CVAE model with the given optimizer and loss function.

        - mse_kl_loss(self, y_true, y_pred):
          Computes the combined Mean Squared Error (MSE) and Kullback-Leibler (KL) loss.

        - train_cvae(self, train_data, epochs, batch_size, callbacks):
          Trains the CVAE model using the provided training data.

    Example Usage:
        # Initialize and build CVAE model
        cvae = CVAE(input_shape=(128, 128, 1), labels=label,
                    config=config)


        # Compile CVAE model
        cvae.compile_cvae(optimizer='adam')

        # Train CVAE model
        cvae.train_cvae(train_data=(x_train, y_train), epochs=1, batch_size=32,
                        callbacks=[tensorboard_callback])
    """

    def __init__(
        self, input_shape: Tuple[float, float], label: float, config: dict
    ):
        self.input_shape = input_shape
        self.label_shape = label_shape
        self.config = config
        self.vae = self.build_cvae()

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampler([z_mean, z_log_var])
        x_decoded_img = self.decoder(z)

        return x_decoded_img, z_mean, z_log_var

    def build_cvae(self) -> Model:
        """
        Builds the CVAE model architecture using the specified hyperparameters.

        This method constructs the encoder and decoder parts of the CVAE,
        and combines them into a single model.

        Returns:
            The constructed CVAE model.
        """
        logging.debug("Building CVAE model using specified hyperparameters.")

        # Encoder Inputs (mask)
        encoder_inputs = Input(shape=self.input_shape)
        logging.debug(f"Encoder: Input shape: {encoder_inputs.shape}")
        x = Reshape((self.input_shape[0], self.input_shape[1], 1))(encoder_inputs)
        x = self.conv_block(
            x,
            self.config["CVAE"]["ref_filters"],
            self.config["CVAE"]["w_init"],
        )
        logging.debug(f"Encoder: Output shape after conv_block: {x.shape}")
        x = Flatten()(x)
        logging.debug(f"Encoder: Output shape after Flatten: {x.shape}")
        x = Dense(64, activation="relu")(x)
        logging.debug(f"Encoder: Output shape after Dense layer: {x.shape}")

        # Latent Space: Mean and Variance, size = latent_dim
        z_mean = Dense(self.config["CVAE"]["latent_dim"], name="z_mean")(x)
        z_log_var = Dense(self.config["CVAE"]["latent_dim"], name="z_log_var")(x)

        # sample z from the latent space
        z = Lambda(
            self.sampler,
            output_shape=(self.config["CVAE"]["latent_dim"],),
            name="z",
        )([z_mean, z_log_var])
        logging.debug(f"Encoder: Output shape after sampler layer: {z.shape}")
        # Encoder Model: mask -> z
        encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        # latent_inputs = Input(
        #     shape=(self.config["CVAE"]["latent_dim"],), name="z_sampling"
        # )
        # label_inputs = Input(shape=(len(self.labels[0]),), name="label")

        # Decoder Inputs (z, label)
        decoder_inputs = Concatenate()([z, self.label])
        logging.debug(f"Decoder: Input shape after Concatenate: {decoder_inputs.shape}")
        x = Dense(64 * 64 * 64, activation="relu")(decoder_inputs)
        logging.debug(f"Decoder: Output shape after Dense layer: {x.shape}")
        x = Reshape((128, 128, 16))(x)
        logging.debug(f"Decoder: Output shape after Reshape layer: {x.shape}")
        x = self.deconv_block(
            x, self.config["CVAE"]["ref_filters"] * 2, self.config["CVAE"]["w_init"]
        )
        logging.debug(f"Decoder: Output shape after deconv_block: {x.shape}")
        x = self.deconv_block(
            x, self.config["CVAE"]["ref_filters"] * 4, self.config["CVAE"]["w_init"]
        )
        logging.debug(f"Decoder: Output shape after deconv_block: {x.shape}")
        x = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)  # Try Tanh?s
        logging.debug(f"Decoder: Output shape after Conv2DTranspose layer: {x.shape}")
        decoder_output = Reshape(self.input_shape)(x)

        # Decoder Model (z, label) -> reconstructed image
        decoder = Model(decoder_inputs, decoder_output, name="decoder")

        # Combined Model
        outputs = decoder([encoder(encoder_inputs)[2], tf.expand_dims(self.labels, axis=0)])
        logging.debug(f"Combined Model: Output shape after decoder: {outputs.shape}")
        vae = Model([encoder_inputs, label_inputs], outputs, name="cvae")
        return vae

    def conv_block(self, input_data, filters, f_init="he_normal"):
        """
        Defines a convolutional block for the encoder part of the CVAE.

        Parameters:
            input_data: The input data for the convolutional block.
            filters: The number of filters in the convolutional block.
            f_init: The initializer for the filters (default is "he_normal").

        Returns:
            The output of the convolutional block.
        """
        x = SeparableConv2D(
            filters,
            kernel_size=(4, 4),
            depthwise_initializer=f_init,
            pointwise_initializer=f_init,
            padding="same",
        )(input_data)
        x = Activation(tf.nn.relu)(x)

        x = SeparableConv2D(
            filters,
            kernel_size=(4, 4),
            depthwise_initializer=f_init,
            pointwise_initializer=f_init,
            padding="same",
        )(x)
        output = Activation(tf.nn.relu)(x)

        m_pool = MaxPooling2D(
            pool_size=(2, 2), strides=2, data_format="channels_last", padding="same"
        )(output)
        return m_pool

    def deconv_block(self, input_data, filters, f_init="he_normal"):
        """
        Defines a deconvolutional block for the decoder part of the CVAE.

        Parameters:
            input_data: The input data for the deconvolutional block.
            filters: The number of filters in the deconvolutional block.
            f_init: The initializer for the filters (default is "he_normal").

        Returns:
            The output of the deconvolutional block.
        """
        x = Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=2,
            kernel_initializer=f_init,
            data_format="channels_last",
            padding="same",
        )(input_data)

        x = SeparableConv2D(
            filters,
            kernel_size=(4, 4),
            depthwise_initializer=f_init,
            pointwise_initializer=f_init,
            padding="same",
        )(x)
        x = Activation(tf.nn.relu)(x)

        x = SeparableConv2D(
            filters,
            kernel_size=(4, 4),
            depthwise_initializer=f_init,
            pointwise_initializer=f_init,
            padding="same",
        )(x)
        output = Activation(tf.nn.relu)(x)
        return output

    def sampler(self, args):
        """
        Sampling function for the latent space during training.

        Parameters:
            args: A tuple containing the mean and log variance of the latent space.

        Returns:
            The sampled latent space.
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    # def mse_kl_loss(self, y_true, y_pred, z_mean, z_log_var):
    #     """
    #     Computes the combined Mean Squared Error (MSE) and Kullback-Leibler (KL) loss.

    #     Parameters:
    #         y_true: The true output values.
    #         y_pred: The predicted output values.
    #         z_mean: The mean of the latent space.
    #         z_log_var: The log variance of the latent space.

    #     Returns:
    #         The combined MSE and KL loss.
    #     """
    #     squared_difference = tf.square(y_true - y_pred)
    #     reconstruction = tf.reduce_mean(squared_difference, axis=-1)
    #     kl_divergence = 0.5 * tf.reduce_sum(
    #         tf.exp(z_log_var) + tf.square(z_mean) - 1.0 - z_log_var, axis=-1
    #     )
    #     return reconstruction + kl_divergence

    # def mse_kl_loss(y_true, y_pred):
    #     """
    #     Computes the combined Mean Squared Error (MSE) and Kullback-Leibler (KL) loss.

    #     Parameters:
    #         y_true: The true output values.
    #         y_pred: The predicted output values.

    #     Returns:
    #         The combined MSE and KL loss.
    #     """
    #     z_mean, z_log_var = model.get_layer("encoder").get_output_at(0)

    #     squared_difference = tf.square(y_true - y_pred)
    #     reconstruction = tf.reduce_mean(squared_difference, axis=-1)
    #     kl_divergence = 0.5 * tf.reduce_sum(
    #         tf.exp(z_log_var) + tf.square(z_mean) - 1.0 - z_log_var, axis=-1
    #     )

    # return reconstruction + kl_divergence

    def mse_kl_loss(model, y_true, y_pred):
        """
        Computes the combined Mean Squared Error (MSE) and Kullback-Leibler (KL) loss.

        Parameters:
            model: The CVAE model instance.
            y_true: The true output values.
            y_pred: The predicted output values.

        Returns:
            The combined MSE and KL loss.
        """
        z_mean, z_log_var = model.get_layer("encoder").get_output_at(0)

        squared_difference = tf.square(y_true - y_pred)
        reconstruction = tf.reduce_mean(squared_difference, axis=-1)
        kl_divergence = 0.5 * tf.reduce_sum(
            tf.exp(z_log_var) + tf.square(z_mean) - 1.0 - z_log_var, axis=-1
        )

        return reconstruction + kl_divergence

    def mse_kl_loss(self, y_true, y_pred):
        """
        Computes the combined Mean Squared Error (MSE) and Kullback-Leibler (KL) loss.

        Parameters:
            y_true: The true output values.
            y_pred: The predicted output values.

        Returns:
            The combined MSE and KL loss.
        """
        encoder_output = self.vae.get_layer("encoder")(y_true)
        z_mean, z_log_var, _ = encoder_output

        squared_difference = tf.square(y_true - y_pred)
        reconstruction = tf.reduce_mean(squared_difference, axis=-1)
        kl_divergence = 0.5 * tf.reduce_sum(
            tf.exp(z_log_var) + tf.square(z_mean) - 1.0 - z_log_var, axis=-1
        )

        return reconstruction + kl_divergence

    # def compile_cvae(
    #     self,
    #     optimizer: str,
    #     lr: float = None,
    #     loss=mse_kl_loss,
    #     metrics: Optional[List[str]] = None,
    # ):
    #     """
    #     Compiles the CVAE model with the given optimizer and loss function.

    #     Parameters:
    #         optimizer: The optimizer to use for training the model.
    #         metrics: The list of metrics to be evaluated during training and testing.
    #     """
    #     if lr is None:
    #         lr = self.config["CVAE"]["lr"]

    #     if metrics is None:
    #         metrics = []

    #     if optimizer.lower() == "adam":
    #         opt = Adam(learning_rate=lr)
    #     else:
    #         raise ValueError(f"Optimizer {optimizer} not supported")

    #     self.vae.compile(optimizer=opt, loss=loss, metrics=metrics)

    def compile_cvae(
        self,
        optimizer: str,
        lr: float = None,
        loss=mse_kl_loss,
        metrics: Optional[List[str]] = None,
    ):
        """
        Compiles the CVAE model with the given optimizer and loss function.

        Parameters:
            optimizer: The optimizer to use for training the model.
            metrics: The list of metrics to be evaluated during training and testing.
        """
        if lr is None:
            lr = self.config["CVAE"]["lr"]

        if metrics is None:
            metrics = []

        if optimizer.lower() == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            raise ValueError(f"Optimizer {optimizer} not supported")

        self.vae.compile(
            optimizer=opt,
            loss=self.mse_kl_loss,  # Use the function directly
            metrics=metrics,
        )

    # def compile_cvae(
    #     self,
    #     optimizer: str,
    #     lr: float = None,
    #     loss=mse_kl_loss,
    #     metrics: Optional[List[str]] = None,
    # ):
    #     """
    #     Compiles the CVAE model with the given optimizer and loss function.

    #     Parameters:
    #         optimizer: The optimizer to use for training the model.
    #         metrics: The list of metrics to be evaluated during training and testing.
    #     """
    #     if lr is None:
    #         lr = self.config["CVAE"]["lr"]

    #     if metrics is None:
    #         metrics = []

    #     if optimizer.lower() == "adam":
    #         opt = Adam(learning_rate=lr)
    #     else:
    #         raise ValueError(f"Optimizer {optimizer} not supported")

    #     # Extract y_true, z_mean, and z_log_var from train_data and validation_data
    #     def extract_loss_args(train_data, validation_data):
    #         x_train, y_train = train_data
    #         x_val, y_val = validation_data
    #         y_pred = self.vae(x_train)
    #         z_mean, z_log_var = self.encoder(x_train)
    #         return y_pred, z_mean, z_log_var, y_val

    #     loss = lambda y_true, y_pred, z_mean, z_log_var: loss(
    #         y_true, y_pred, z_mean, z_log_var
    #     )

    #     self.vae.compile(optimizer=opt, loss=loss, metrics=metrics)

    def train(
        self,
        train_data: Tuple,
        epochs: int,
        callbacks: Optional[List] = [],
        batch_size: int = 1,
        validation_data: Optional[Tuple] = None,
    ):
        """
        Trains the Conditional Variational Autoencoder (CVAE) model using the provided training data.

        Parameters:
            train_data (Tuple): A tuple containing the input data and labels for the model.
                - x_train (numpy.ndarray): The input data for the model. It should be a numpy array
                    with the shape (num_samples, height, width, channels).
                - y_train (numpy.ndarray): The training labels for the model. It should be a numpy
                    array with the shape (num_samples, size_label).

            epochs (int): The number of epochs to train the model. An epoch is a complete pass
                          through the entire training dataset.
            batch_size (int): The size of the batches to use for training. A batch is a subset of
                             the training dataset used to compute the gradient and update the model's weights.
            callbacks (List): A list of callbacks to be called during training. Callbacks are used to
                              monitor training and perform custom actions, such as saving the model
                              after each epoch or stopping training if the loss does not improve.

        Returns:
            history (History): A History object containing the training history of the model. This
            object contains the loss and any additional metrics specified during the model compilation.
        """
        x_train, y_train = train_data
        x_val, y_val = validation_data
        history = self.vae.fit(
            [x_train, y_train],
            x_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data=([x_val, y_val], x_val),
        )
        return history


# --------------------------------------------------------------

# import tensorflow as tf


# # processes input image and flattens feature maps
# def get_conditional_encoder1():
#     inputs = tf.keras.Input(shape=(28, 28, 1))
#     x = tf.keras.layers.Conv2D(
#         filters=32, kernel_size=3, strides=(2, 2), activation="relu"
#     )(inputs)
#     x = tf.keras.layers.Conv2D(
#         filters=64, kernel_size=3, strides=(2, 2), activation="relu"
#     )(x)
#     x = tf.keras.layers.Flatten()(x)

#     return tf.keras.Model(inputs=inputs, outputs=[x])


# # gets flattened feature maps, and one hot label vector and outputs mu and rho
# def get_conditional_encoder2(latent_dim, input_size):
#     inputs = tf.keras.Input(shape=(input_size + 10,))
#     mu = tf.keras.layers.Dense(units=latent_dim)(inputs)
#     rho = tf.keras.layers.Dense(units=latent_dim)(inputs)

#     return tf.keras.Model(inputs=inputs, outputs=[mu, rho])


# # classical vae decoder
# def get_conditional_decoder(latent_dim):
#     z = tf.keras.Input(shape=(latent_dim + 10,))
#     x = tf.keras.layers.Dense(units=7 * 7 * 32, activation="relu")(z)
#     x = tf.keras.layers.Reshape(target_shape=(7, 7, 32))(x)
#     x = tf.keras.layers.Conv2DTranspose(
#         filters=64, kernel_size=3, strides=2, padding="same", activation="relu"
#     )(x)
#     x = tf.keras.layers.Conv2DTranspose(
#         filters=32, kernel_size=3, strides=2, padding="same", activation="relu"
#     )(x)
#     decoded_img = tf.keras.layers.Conv2DTranspose(
#         filters=1, kernel_size=3, strides=1, padding="same"
#     )(x)

#     return tf.keras.Model(inputs=z, outputs=[decoded_img])


# class Conditional_VAE(tf.keras.Model):
#     def __init__(self, latent_dim):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.encoder_block1 = get_conditional_encoder1()
#         # 2304 is specific to conv layers, not the best practice to hardcode it
#         self.encoder_block2 = get_conditional_encoder2(
#             latent_dim=latent_dim, input_size=2304
#         )
#         self.decoder_block = get_conditional_decoder(latent_dim)

#     def call(self, img, labels):
#         # encoder q(z|x,y)
#         enc1_output = self.encoder_block1(img)
#         # concat feature maps and one hot label vector
#         img_lbl_concat = np.concatenate((enc1_output, labels), axis=1)
#         z_mu, z_rho = self.encoder_block2(img_lbl_concat)

#         # sampling
#         epsilon = tf.random.normal(shape=z_mu.shape, mean=0.0, stddev=1.0)
#         z = z_mu + tf.math.softplus(z_rho) * epsilon

#         # decoder p(x|z,y)
#         z_lbl_concat = np.concatenate((z, labels), axis=1)
#         decoded_img = self.decoder_block(z_lbl_concat)

#         return z_mu, z_rho, decoded_img
