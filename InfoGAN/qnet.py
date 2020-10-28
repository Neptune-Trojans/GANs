import tensorflow as tf

from InfoGAN.discriminator import Discriminator
from InfoGAN.generator import Generator


class QNet(tf.keras.Model):
    def __init__(self):
        super(QNet, self).__init__()

        self.Qd = tf.keras.layers.Dense(128)
        self.Qb = tf.keras.layers.BatchNormalization()
        self.Qa = tf.keras.layers.LeakyReLU()

        self.Q_cat = tf.keras.layers.Dense(10)
        self.Q_con1_mu = tf.keras.layers.Dense(2)
        self.Q_con1_var = tf.keras.layers.Dense(2)
        self.Q_con2_mu = tf.keras.layers.Dense(2)
        self.Q_con2_var = tf.keras.layers.Dense(2)

    def sample(self, mu, var):
        eps = tf.random.normal(shape=mu.shape)
        sigma = tf.sqrt(var)
        z = mu + sigma * eps

        return z

    def call(self, x, training=True):
        q = self.Qd(x)
        q = self.Qb(x, training=training)
        q = self.Qa(x)

        Q_cat = self.Q_cat(q)

        Q_con1_mu = self.Q_con1_mu(q)
        Q_con1_var = tf.exp(self.Q_con1_var(q))
        Q_con2_mu = self.Q_con2_mu(q)
        Q_con2_var = tf.exp(self.Q_con2_var(q))

        Q_con1 = self.sample(Q_con1_mu, Q_con1_var)
        Q_con2 = self.sample(Q_con2_mu, Q_con2_var)

        return Q_cat, Q_con1, Q_con2


if __name__ == "__main__":
    import numpy as np

    # tf.debugging.set_log_device_placement(True)
    z = np.random.normal(size=(1, 74)).astype(np.float32)
    z = tf.convert_to_tensor(z)

    g = Generator()
    d = Discriminator()
    image = g(z)
    prediction = d(image)
    print(prediction[0])
    print(prediction[1])