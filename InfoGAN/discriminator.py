import tensorflow as tf


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.c1 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same")
        self.a1 = tf.keras.layers.LeakyReLU()

        self.c2 = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same")
        self.a2 = tf.keras.layers.LeakyReLU()
        self.b2 = tf.keras.layers.BatchNormalization()
        self.f2 = tf.keras.layers.Flatten()

        self.d3 = tf.keras.layers.Dense(1024)
        self.a3 = tf.keras.layers.LeakyReLU()
        self.b3 = tf.keras.layers.BatchNormalization()

        self.D = tf.keras.layers.Dense(1)

    def call(self, x, training=True):
        x = self.c1(x)
        x = self.a1(x)

        x = self.c2(x)
        x = self.b2(x, training=training)
        x = self.a2(x)
        x = self.f2(x)

        x = self.d3(x)
        x = self.b3(x, training=training)
        x = self.a3(x)

        mid = x

        D = self.D(x)

        return D, mid