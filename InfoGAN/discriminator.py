import tensorflow as tf
from tensorflow.keras import layers


class Discriminator:
    def __init__(self):
        pass

    @staticmethod
    def make_discriminator_model(n_class=10):
        inputs = tf.keras.layers.Input(shape=(28, 28, 1))

        x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(inputs)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)

        flatten = layers.Flatten()(x)
        o = layers.Dense(1)(flatten)
        # TODO Add activation ? change model little bit ?
        disc_model = tf.keras.models.Model(inputs=inputs, outputs=[o])

        q = layers.Dense(1000)(flatten)
        q = layers.Dense(100)(q)

        # Gaussian distribution mean (continuous output)
        mu = layers.Dense(1)(q)

        # Gaussian distribution standard deviation (exponential activation to ensure the value is positive)
        sigma = layers.Dense(1, activation=lambda x: tf.math.exp(x))(q)

        # Classification (discrete output)
        clf_out = layers.Dense(n_class, activation="softmax")(q)

        q_model = tf.keras.models.Model(inputs=inputs, outputs=[clf_out, mu, sigma])

        return disc_model, q_model
