#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from typing import Tuple
import logging

logging.basicConfig(level=logging.DEBUG)


# closed form kl loss computation between variational posterior q(z|x) and unit Gaussian prior p(z)
def kl_loss(z_mu, z_rho):
    sigma_squared = tf.math.softplus(z_rho) ** 2
    kl_1d = -0.5 * (1 + tf.math.log(sigma_squared) - z_mu**2 - sigma_squared)

    # sum over sample dim, average over batch dim
    kl_batch = tf.reduce_mean(tf.reduce_sum(kl_1d, axis=1))

    return kl_batch


def elbo(z_mu, z_rho, decoded_img, original_img):
    # reconstruction loss
    mse = tf.reduce_mean(tf.reduce_sum(tf.square(original_img - decoded_img), axis=1))
    # kl loss
    kl = kl_loss(z_mu, z_rho)

    return mse, kl


class CVAE(tf.keras.Model):
    """
    Conditional Variational Autoencoder for continuous data.
    """

    def __init__(self, input_dim: Tuple[int, int, int], latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def conv_block(
        self, filters: int = 32, f_init: str = "he_normal"
    ) -> tf.keras.Sequential:
        """
        Defines a convolutional block for the encoder part of the CVAE.

        Parameters:
            filters: The number of filters in the convolutional block.
            f_init: The initializer for the filters (default is "he_normal").

        Returns:
            A Sequential model representing the convolutional block.
        """
        return tf.keras.Sequential(
            [
                tf.keras.layers.SeparableConv2D(
                    filters,
                    kernel_size=(4, 4),
                    depthwise_initializer=f_init,
                    pointwise_initializer=f_init,
                    padding="same",
                    activation=tf.nn.relu,
                ),
                tf.keras.layers.SeparableConv2D(
                    filters,
                    kernel_size=(4, 4),
                    depthwise_initializer=f_init,
                    pointwise_initializer=f_init,
                    padding="same",
                    activation=tf.nn.relu,
                ),
                tf.keras.layers.MaxPooling2D(
                    pool_size=(2, 2),
                    strides=2,
                    data_format="channels_last",
                    padding="same",
                ),
            ]
        )

    def deconv_block(self, filters, f_init="he_normal"):
        """
        Defines a deconvolutional block for the decoder part of the CVAE.

        Parameters:
            filters: The number of filters in the deconvolutional block.
            f_init: The initializer for the filters (default is "he_normal").

        Returns:
            A Sequential model representing the deconvolutional block.
        """
        return tf.keras.Sequential(
            [
                tf.keras.layers.Conv2DTranspose(
                    filters,
                    kernel_size=(4, 4),
                    strides=2,
                    kernel_initializer=f_init,
                    data_format="channels_last",
                    padding="same",
                ),
                tf.keras.layers.SeparableConv2D(
                    filters,
                    kernel_size=(4, 4),
                    depthwise_initializer=f_init,
                    pointwise_initializer=f_init,
                    padding="same",
                    activation=tf.nn.relu,
                ),
                tf.keras.layers.SeparableConv2D(
                    filters,
                    kernel_size=(4, 4),
                    depthwise_initializer=f_init,
                    pointwise_initializer=f_init,
                    padding="same",
                    activation=tf.nn.relu,
                ),
            ]
        )

    def build_encoder(self):
        conv_block = tf.keras.Sequential(
            [
                tf.keras.layers.SeparableConv2D(
                    filters=128,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    activation="relu",
                    padding="same",
                ),
                tf.keras.layers.SeparableConv2D(
                    filters=128,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    activation="relu",
                    padding="same",
                ),
            ]
        )

        encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=self.input_dim),
                conv_block,
                conv_block,
                # to reduce the number of parameters
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(self.latent_dim + self.latent_dim),
            ]
        )
        return encoder

    def build_decoder(self):
        decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="relu",
                ),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding="same"
                ),
            ]
        )
        return decoder

    def call(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        z_mean, z_log_var = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        z = self.reparameterize(z_mean, z_log_var)
        return self.decoder(z)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean
