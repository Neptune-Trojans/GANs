import tensorflow as tf


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.d1 = tf.keras.layers.Dense(1024, use_bias=False)
        self.a1 = tf.keras.layers.ReLU()
        self.b1 = tf.keras.layers.BatchNormalization()
        self.d2 = tf.keras.layers.Dense(7 * 7 * 128, use_bias=False)
        self.a2 = tf.keras.layers.ReLU()
        self.b2 = tf.keras.layers.BatchNormalization()
        self.r2 = tf.keras.layers.Reshape([7, 7, 128])

        self.c3 = tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same")
        self.a3 = tf.keras.layers.ReLU()
        self.b3 = tf.keras.layers.BatchNormalization()

        self.c4 = tf.keras.layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding="same")

    def call(self, x, training=True):
        x = self.d1(x)
        x = self.b1(x, training=training)
        x = self.a1(x)

        x = self.d2(x)
        x = self.b2(x, training=training)
        x = self.a2(x)
        x = self.r2(x)

        x = self.c3(x)
        x = self.b3(x, training=training)
        x = self.a3(x)

        x = self.c4(x)

        x = tf.nn.tanh(x)

        return x