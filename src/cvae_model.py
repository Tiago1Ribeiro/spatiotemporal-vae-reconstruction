# 
import os
import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2DTranspose
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, Activation
from tensorflow.keras.layers import Lambda, Concatenate, Flatten
from tensorflow.keras.models import Model


class CVAE:
    """
    
    
    """    
    def __init__(self, input_shape, num_labels, train_labels, config):
        self.input_shape = input_shape
        self.num_labels = num_labels
        self.train_labels = train_labels
        self.config = config
        self.vae = self.build_cvae()

    def build_cvae(self):
        encoder_inputs = Input(shape=self.input_shape)
        x = Reshape((self.input_shape[0],
                     self.input_shape[1],
                     1))(encoder_inputs)
        x = self.conv_block(x,
                            self.config["models"]["CVAE"]["ref_filters"],
                            self.config["models"]["CVAE"]["w_init"])
        x = Flatten()(x)
        x = Dense(64, activation="relu")(x)

        z_mean = Dense(self.config["models"]["CVAE"]["latent_dim"],
                       name="z_mean")(x)
        z_log_var = Dense(self.config["models"]["CVAE"]["latent_dim"], 
                          name="z_log_var")(x)
        z = Lambda(self.sampler, output_shape=(self.config["models"]["CVAE"]["latent_dim"],), name='z')([z_mean, z_log_var])

        encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        latent_inputs = Input(shape=(self.config["models"]["CVAE"]["latent_dim"],), name='z_sampling')
        label_inputs = Input(shape=(self.num_labels,), name='label')
        decoder_inputs = Concatenate()([latent_inputs, label_inputs])
        x = Dense(64*64*64, activation="relu")(decoder_inputs)
        x = Reshape((128, 128, 16))(x)
        x = self.deconv_block(x, self.args["REF_FILTERS"]*2, self.args["F_INIT"])
        x = self.deconv_block(x, self.args["REF_FILTERS"]*4, self.args["F_INIT"])
        x = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        decoder_output = Reshape(self.input_shape)(x)
        decoder = Model([latent_inputs, label_inputs], decoder_output, name="decoder")

        outputs = decoder([encoder(encoder_inputs)[2], label_inputs])
        vae = Model([encoder_inputs, label_inputs], outputs, name='cvae')
        return vae

    def conv_block(self, input_data, filters, f_init="he_normal"):
        x = SeparableConv2D(filters,
                            kernel_size=(4, 4),
                            depthwise_initializer=f_init,
                            pointwise_initializer=f_init,
                            padding="same")(input_data)
        x = Activation(tf.nn.relu)(x)

        x = SeparableConv2D(filters,
                            kernel_size=(4, 4),
                            depthwise_initializer=f_init,
                            pointwise_initializer=f_init,
                            padding="same")(x)
        output = Activation(tf.nn.relu)(x)

        m_pool = MaxPooling2D(pool_size=(2, 2),
                              strides=2,
                              data_format="channels_last",
                              padding='same')(output)
        return m_pool
    
    
        encoder_inputs = Input(shape=self.input_shape)
        x = Reshape((self.config["data"]["input_size"][0],
                     self.config["data"]["input_size"][1],
                     1))(encoder_inputs)
        x = self.conv_block(x,
                            self.config["models"]["CVAE"]["ref_filters"],
                            self.config["models"]["CVAE"]["w_init"])
        x = Flatten()(x)
        x = Dense(64, activation="relu")(x)

        z_mean = Dense(self.config["models"]["CVAE"]["latent_dim"],
                       name="z_mean")(x)
        z_log_var = Dense(self.args["LATENT_DIM"], name="z_log_var")(x)
        z = Lambda(self.sampler, output_shape=(self.args["LATENT_DIM"],), name='z')([z_mean, z_log_var])

        encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        latent_inputs = Input(shape=(self.args["LATENT_DIM"],), name='z_sampling')
        label_inputs = Input(shape=(self.num_labels,), name='label')
        decoder_inputs = Concatenate()([latent_inputs, label_inputs])
        x = Dense(64*64*64, activation="relu")(decoder_inputs)
        x = Reshape((128, 128, 16))(x)
        x = self.deconv_block(x, self.args["REF_FILTERS"]*2, self.args["F_INIT"])
        x = self.deconv_block(x, self.args["REF_FILTERS"]*4, self.args["F_INIT"])
        x = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        decoder_output = Reshape(self.input_shape)(x)
        decoder = Model([latent_inputs, label_inputs], decoder_output, name="decoder")

        outputs = decoder([encoder(encoder_inputs)[2], label_inputs])
        vae = Model([encoder_inputs, label_inputs], outputs, name='cvae')
        return vae

    def conv_block(self, input_data, filters, f_init="he_normal"):
        x = SeparableConv2D(filters,
                            kernel_size=(4, 4),
                            depthwise_initializer=f_init,
                            pointwise_initializer=f_init,
                            padding="same")(input_data)
        x = Activation(tf.nn.relu)(x)

        x = SeparableConv2D(filters,
                            kernel_size=(4, 4),
                            depthwise_initializer=f_init,
                            pointwise_initializer=f_init,
                            padding="same")(x)
        output = Activation(tf.nn.relu)(x)

        m_pool = MaxPooling2D(pool_size=(2, 2),
                              strides=2,
                              data_format="channels_last",
                              padding='same')(output)
        return m_pool

    def deconv_block(self, input_data, filters, f_init="he_normal"):
        x = Conv2DTranspose(filters,
                            kernel_size=(4, 4),
                            strides=2,
                            kernel_initializer=f_init,
                            data_format="channels_last",
                            padding="same")(input_data)

        x = SeparableConv2D(filters,
                            kernel_size=(4, 4),
                            depthwise_initializer=f_init,
                            pointwise_initializer=f_init,
                            padding="same")(x)
        x = Activation(tf.nn.relu)(x)

        x = SeparableConv2D(filters,
                            kernel_size=(4, 4),
                            depthwise_initializer=f_init,
                            pointwise_initializer=f_init,
                            padding="same")(x)
        output = Activation(tf.nn.relu)(x)
        return output

    def sampler(self, args):
       z_mean, z_log_var = args
       batch = K.shape(z_mean)[0]
       dim = K.shape(z_mean)[1]
       epsilon = K.random_normal(shape=(batch, dim))
       return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def compile_cvae(self, optimizer):
       self.vae.compile(optimizer=optimizer, loss=self.mse_kl_loss)

    def mse_kl_loss(self, y_true, y_pred):
       squared_difference = tf.square(y_true - y_pred)
       reconstruction = tf.reduce_mean(squared_difference, axis=-1)
       kl_divergence = 0.5 * tf.reduce_sum(tf.exp(z_log_var) + tf.square(z_mean) - 1. - z_log_var, axis=-1)
       return reconstruction + kl_divergence

    def train_cvae(self, train_data, epochs, batch_size, callbacks):
       # Assuming train_data is a tuple (x_train, y_train)
       x_train, y_train = train_data
       self.vae.fit([x_train, y_train],
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks)
