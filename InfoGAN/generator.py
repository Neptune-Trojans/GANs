import tensorflow as tf
from tensorflow.keras import layers


class Generator:

    def __init__(self):
        pass

    @staticmethod
    def make_generator_model():
        inputs = tf.keras.layers.Input(shape=(100,))

        x = layers.Dense(7 * 7 * 256, use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Reshape((7, 7, 256))(x)

        x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)

        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)

        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=[x])

        return model
