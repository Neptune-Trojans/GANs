import os
import shutil

import tensorflow as tf
import tensorflow_probability as tfp

from InfoGAN.discriminator import Discriminator
from InfoGAN.generator import Generator
from InfoGAN.mnist_data import load_data_mnist
from InfoGAN.training_arguments import Arguments
from InfoGAN.training_visualization import Visualization


class Trainer:
    def __init__(self, args):
        self._args = args
        self._data_gen = load_data_mnist(self._args.buffer_size, self._args.batch_size)
        self._disc_model, self._q_model = Discriminator.make_discriminator_model()
        self._gen_model = Generator.make_generator_model()
        self._disc_optimizer = self._create_optimizer(init_lr=2e-4)
        self._gen_optimizer = self._create_optimizer(init_lr=5e-4)
        self._q_optimizer = self._create_optimizer(init_lr=2e-4)
        self._vis_label, self._vis_cat, self._vis_noise = self.create_gen_input(self._args.num_examples_to_generate, self._args.noise_dim, self._args.num_classes)

        self._visualization = Visualization(self._args.visualization_folder, self._vis_label, self._vis_cat, self._vis_noise)
        self.categorical_loss = tf.keras.losses.CategoricalCrossentropy()
        self._generate_folders([self._args.visualization_folder, self._args.check_points])

    @staticmethod
    def _generate_folders(folders):
        for folder in folders:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder)

    @staticmethod
    def _create_optimizer(init_lr: float, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-8):
        optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr, beta_1=adam_beta1, beta_2=adam_beta2, epsilon=adam_epsilon)
        return optimizer

    @staticmethod
    def discriminator_loss(real, fake):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real), real)
        fake_loss = cross_entropy(tf.zeros_like(fake), fake)

        total_loss = real_loss + fake_loss

        return total_loss

    @staticmethod
    def generator_loss(fake):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss = cross_entropy(tf.ones_like(fake), fake)

        return loss

    def create_gen_input(self, batch_size, noise_size, n_class):
        # create noise input
        noise = tf.random.normal([batch_size, noise_size])
        # Create categorical latent code
        label = tf.random.uniform([batch_size], minval=0, maxval=10, dtype=tf.int32)
        label = tf.one_hot(label, depth=n_class)
        # Create one continuous latent code
        c_1 = tf.random.uniform([batch_size, 1], minval=-1, maxval=1)

        return label, c_1, noise

    def train(self):

        generator_loss = tf.keras.metrics.Mean(name='gen_loss')
        discriminator_loss = tf.keras.metrics.Mean(name='discriminator_loss')

        categorical_loss = tf.keras.metrics.Mean(name='categorical_loss')
        continuous_loss = tf.keras.metrics.Mean(name='continuous_loss')

        @tf.function
        def train_step(images, batch_size, noise_dim, num_classes):
            # Create generator input
            label, c_1, noise = self.create_gen_input(batch_size // 2, noise_dim, num_classes)
            gen_input = tf.keras.layers.Concatenate()([label, c_1, noise])
            # training only discriminator
            with tf.GradientTape() as d_tape:
                # should be trainable
                self._disc_model.trainable = True
                d_tape.watch(self._disc_model.trainable_variables)
                # TODO should be used first half of images
                # Train discriminator using real images
                d_real_output = self._disc_model(images, training=True)

                # Fake images part
                fake_image_batch = self._gen_model(gen_input, training=False)
                d_fake_output = self._disc_model(fake_image_batch, training=True)

                d_loss = self.discriminator_loss(d_real_output, d_fake_output)

            d_gradients = d_tape.gradient(d_loss, self._disc_model.trainable_variables)
            self._disc_optimizer.apply_gradients(zip(d_gradients, self._disc_model.trainable_variables))

            # training generator and q networks
            # We do not want to modify the neurons in the discriminator when training the generator and the auxiliary model
            self._disc_model.trainable = False
            label, c_1, noise = self.create_gen_input(batch_size, noise_dim, num_classes)
            gen_input = tf.keras.layers.Concatenate()([label, c_1, noise])
            with tf.GradientTape() as g_tape, tf.GradientTape() as q_tape:

                g_tape.watch(self._gen_model.trainable_variables)
                q_tape.watch(self._q_model.trainable_variables)

                # Create fake image batch
                fake_image_batch = self._gen_model(gen_input, training=True)
                d_fake_output = self._disc_model(fake_image_batch, training=False)
                gen_loss = self.generator_loss(d_fake_output)

                # Auxiliary loss
                cat_output, mu, sigma = self._q_model(fake_image_batch, training=True)
                # Categorical loss
                cat_loss = self.categorical_loss(label, cat_output)
                # Use Gaussian distributions to represent the output
                dist = tfp.distributions.Normal(loc=mu, scale=sigma)
                # Losses (negative log probability density function as we want to maximize the probability density function)
                c_1_loss = tf.reduce_mean(-dist.log_prob(c_1))
                # Generator total loss
                g_loss = gen_loss + (cat_loss + 0.1 * c_1_loss)
                # Auxiliary function loss
                q_loss = (cat_loss + 0.1 * c_1_loss)
            # Calculate gradients

            g_gradients = g_tape.gradient(g_loss, self._gen_model.trainable_variables)
            q_gradients = q_tape.gradient(q_loss, self._q_model.trainable_variables)
            # Optimize
            self._gen_optimizer.apply_gradients(zip(g_gradients, self._gen_model.trainable_variables))
            self._q_optimizer.apply_gradients(zip(q_gradients, self._q_model.trainable_variables))

            generator_loss(gen_loss)
            categorical_loss(cat_loss)
            continuous_loss(c_1_loss)
            discriminator_loss(d_loss)

        for epoch in range(self._args.epochs):
            # Reset the metrics at the start of the next epoch
            generator_loss.reset_states()
            discriminator_loss.reset_states()
            categorical_loss.reset_states()
            continuous_loss.reset_states()

            print('Epoch {} Started'.format(epoch))
            for step, image_batch in enumerate(self._data_gen):
                train_step(image_batch, self._args.batch_size, self._args.noise_dim, self._args.num_classes)

            template = 'Epoch {}, ' \
                       'Generator Loss: {},  Discriminator Loss: {}, ' \
                       'categorical_loss {},  continuous_loss  {}'

            print(template.format(epoch,
                                  generator_loss.result(), discriminator_loss.result(),
                                  categorical_loss.result(), continuous_loss.result()))

            self._visualization.save_predicted_images(self._gen_model, self._q_model, epoch)
            if epoch % 5 == 0:
                self._gen_model.save("{}/model_{}.h5".format(self._args.check_points, epoch + 1), save_weights=True)

        self._visualization.generate_gif_image()
        self._gen_model.save("{}/model_{}.h5".format(self._args.check_points, 'final'), save_weights=True)



# writer = tf.summary.create_file_writer("./infogan")
# checkpoint_dir = './ckpt'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=gen_opt,
#                                  discriminator_optimizer=dis_opt,
#                                  generator=generator,
#                                  discriminator=discriminator,
#                                  qnet=qnet)


if __name__ == "__main__":
    a = Arguments()
    t = Trainer(a)
    t.train()