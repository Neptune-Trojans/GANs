import tensorflow as tf

from InfoGAN.training_arguments import Arguments
from InfoGAN.training_visualization import Visualization


class VerifyTrainedModel:
    def __init__(self, trained_model_path, noise_size):
        self._trained_model_path = trained_model_path
        self._gen_model = tf.keras.models.load_model(self._trained_model_path)
        self._noise_size = noise_size

    def categorical_vector(self):
        # create noise input same noise ten times
        # we want to see how change only in categorical input changes the visualization
        noise = tf.random.normal([1, self._noise_size])
        noise = tf.repeat(noise, repeats=10, axis=0)

        # Create categorical latent code 10 classes
        label = tf.range(0, 10, delta=1, dtype=tf.int32, name='range')
        label = tf.one_hot(label, depth=10)
        # Create one continuous latent code that is used in all examples
        c_1 = tf.random.uniform([1, 1], minval=-1, maxval=1)
        c_1 = tf.repeat(c_1, repeats=10, axis=0)

        gen_input = tf.keras.layers.Concatenate()([label, c_1, noise])

        images = self._gen_model(gen_input, training=False)
        Visualization.save_predicted_classes(images, label, c_1, 'trained_model/predicted_classes.png')

    def single_digit_different_continuous(self, digit):
        # create noise input same noise ten times
        # we want to see how change only in categorical input changes the visualization
        noise = tf.random.normal([1, self._noise_size])
        noise = tf.repeat(noise, repeats=10, axis=0)

        # Create categorical latent code 10 classes
        label = digit * tf.ones((10, ), dtype=tf.int32, name=None)
        label = tf.one_hot(label, depth=10)

        # Create one continuous latent code that is used in all examples
        # Create one continuous latent code
        c_1 = tf.random.uniform([10, 1], minval=-1, maxval=1)

        gen_input = tf.keras.layers.Concatenate()([label, c_1, noise])

        images = self._gen_model(gen_input, training=False)
        Visualization.save_predicted_classes(images, label, c_1, 'trained_model/digit_{}.png'.format(digit))



if __name__ == '__main__':
    a = Arguments()
    v = VerifyTrainedModel('trained_model/final.h5', a.noise_dim)
    v.categorical_vector()
    for i in range(0, 9):
        v.single_digit_different_continuous(i)