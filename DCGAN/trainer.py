import tensorflow as tf
from absl import app

from DCGAN.BatchGenerator import BatchDataGenerator
from DCGAN.training_arguments import Arguments
from DCGAN.discriminator import Discriminator
from DCGAN.generator import Generator
from DCGAN.training_visualization import Visualization


class Trainer:
    def __init__(self, arguments, data_generator, generator, discriminator):
        self._arguments = arguments
        self.data_generator = data_generator
        self._cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self._visualization_seed = tf.random.normal([self._arguments.num_examples_to_generate,
                                                     self._arguments.noise_dim])

        self._visualization = Visualization(self._arguments.visualization_folder, self._visualization_seed)
        self.generator = generator
        self.discriminator = discriminator

    @staticmethod
    def _create_optimizer(init_lr: float,
                          adam_beta1: float = 0.9,
                          adam_beta2: float = 0.999,
                          adam_epsilon: float = 1e-8
                          ):

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=init_lr, beta_1=adam_beta1, beta_2=adam_beta2, epsilon=adam_epsilon
        )

        return optimizer

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self._cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self._cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self._cross_entropy(tf.ones_like(fake_output), fake_output)

    def train(self):

        generator_optimizer = self._create_optimizer(self._arguments.init_lr)
        discriminator_optimizer = self._create_optimizer(self._arguments.init_lr)

        generator_loss = tf.keras.metrics.Mean(name='generator_loss')
        discriminator_loss = tf.keras.metrics.Mean(name='generator_loss')

        @tf.function
        def train_step(images, batch_size, noise_dim):
            noise = tf.random.normal([batch_size, noise_dim])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = self.generator(noise, training=True)

                real_output = self.discriminator(images, training=True)
                fake_output = self.discriminator(generated_images, training=True)

                gen_loss = self.generator_loss(fake_output)
                disc_loss = self.discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

            generator_loss(gen_loss)
            discriminator_loss(disc_loss)

        for epoch in range(self._arguments.epochs):
            # Reset the metrics at the start of the next epoch
            generator_loss.reset_states()
            discriminator_loss.reset_states()

            print('Epoch {} Started'.format(epoch))
            for step, image_batch in enumerate(self.data_generator):
                train_step(image_batch, self._arguments.batch_size, self._arguments.noise_dim)

            template = 'Epoch {}, ' \
                       'Generator Loss: {},  Discriminator Loss: {}, ' \
                       'Generator LR {},  Discriminator LR {}'

            print(template.format(epoch,
                                  generator_loss.result(), discriminator_loss.result(),
                                  generator_optimizer._decayed_lr(tf.float32).numpy(),
                                  discriminator_optimizer._decayed_lr(tf.float32).numpy()))

            self._visualization.save_predicted_images(self.generator, epoch)
        self._visualization.generate_gif_image()


def main(_):
    arguments = Arguments()
    train_dataset = BatchDataGenerator.load_data_mnist(arguments.buffer_size, arguments.batch_size)
    generator = Generator.make_generator_model()
    discriminator = Discriminator.make_discriminator_model()

    trainer = Trainer(arguments, train_dataset, generator, discriminator)

    trainer.train()


if __name__ == '__main__':
    app.run(main)
