import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from Helpers.training_visualization import Visualization
from InfoGAN.discriminator import Discriminator
from InfoGAN.generator import Generator
from InfoGAN.mnist_data import load_data_mnist
from InfoGAN.qnet import QNet
from InfoGAN.training_arguments import Arguments


class Trainer:
    def __init__(self, args):
        self._args = args
        self._data_gen = load_data_mnist(self._args.buffer_size, self._args.batch_size)
        self._discriminator = Discriminator()
        self._generator = Generator()
        self._qnet = QNet()
        self._visualization_seed = tf.random.normal([self._args.num_examples_to_generate,
                                                     self._args.noise_dim])

        self._visualization = Visualization(self._args.visualization_folder, self._visualization_seed)

    @staticmethod
    def _create_optimizer(init_lr: float,
                          adam_beta1: float = 0.9,
                          adam_beta2: float = 0.999,
                          adam_epsilon: float = 1e-8
                          ):

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

    def train(self):

        generator_optimizer, generator_lr_schedule = self._create_optimizer(self._args.init_lr)
        discriminator_optimizer, discriminator_lr_schedule = self._create_optimizer(self._args.init_lr)

        generator_loss = tf.keras.metrics.Mean(name='generator_loss')
        discriminator_loss = tf.keras.metrics.Mean(name='generator_loss')

        @tf.function
        def train_step(images, batch_size, noise_dim):
            noise = tf.random.normal([batch_size, noise_dim])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = self._generator(noise, training=True)

                real_output, _ = self._discriminator(images, training=True)
                fake_output, mid = self._discriminator(generated_images, training=True)
                fqcat, fqcon1, fqcon2 = self._qnet(mid)

                info_loss, c1, c2, sce = get_mi(fqcon1, fqcon2, fqcat, z_con1, z_con2, z_cat)

                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

                # gi = gen_loss + info_loss
                # di = disc_loss + info_loss

                # real_output = self._discriminator(images, training=True)
                # fake_output = self._discriminator(generated_images, training=True)
                #
                # gen_loss = self.generator_loss(fake_output)
                # disc_loss = self.discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss + info_loss, self._generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss + info_loss, self._discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, self._generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self._discriminator.trainable_variables))

            generator_loss(gen_loss + info_loss)
            discriminator_loss(disc_loss + info_loss)

        for epoch in range(self._args.epochs):
            # Reset the metrics at the start of the next epoch
            generator_loss.reset_states()
            discriminator_loss.reset_states()

            print('Epoch {} Started'.format(epoch))
            for step, image_batch in enumerate(self._data_gen):
                train_step(image_batch, self._args.batch_size, self._args.noise_dim)

            template = 'Epoch {}, ' \
                       'Generator Loss: {},  Discriminator Loss: {}, ' \
                       'Generator LR {},  Discriminator LR {}'

            print(template.format(epoch,
                                  generator_loss.result(), discriminator_loss.result(),
                                  generator_optimizer._decayed_lr(tf.float32).numpy(),
                                  discriminator_optimizer._decayed_lr(tf.float32).numpy()))

            self._visualization.save_predicted_images(self._generator, epoch)
        self._visualization.generate_gif_image()





# writer = tf.summary.create_file_writer("./infogan")
# checkpoint_dir = './ckpt'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=gen_opt,
#                                  discriminator_optimizer=dis_opt,
#                                  generator=generator,
#                                  discriminator=discriminator,
#                                  qnet=qnet)


# def train_display_image(model, epoch):
#     z1, _, _, _ = sample(4, 0)
#     z2, _, _, _ = sample(4, 1)
#     z3, _, _, _ = sample(4, 2)
#     z4, _, _, _ = sample(4, 3)
#     z = tf.concat([z1, z2, z3, z4], axis=0)
#     predictions = model(z, training=False)
#     predictions = (predictions + 1.) / 2.
#
#     plt.figure(figsize=(4, 4))
#     plt.suptitle(epoch + 1)
#     for i in range(predictions.shape[0]):
#         plt.subplot(4, 4, i + 1)
#         plt.imshow(tf.reshape(predictions[i], [28, 28]), cmap='gray')
#         plt.axis("off")
#
#     plt.savefig('train_images/img_step{:04d}.png'.format(epoch))
#     plt.close()





def get_mi(fqcon1, fqcon2, fqcat, z_con1, z_con2, z_cat):
    c1 = tf.reduce_mean(tf.reduce_sum(tf.square(fqcon1 - z_con1), -1)) * 0.5
    c2 = tf.reduce_mean(tf.reduce_sum(tf.square(fqcon2 - z_con2), -1)) * 0.5
    sce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(z_cat, fqcat)

    info_loss = c1 + c2 + sce

    return info_loss, c1, c2, sce


def sample(size, cat=-1, c1=None, c2=None):
    z = tfd.Uniform(low=-1.0, high=1.0).sample([size, 62])

    if c1 is not None:
        z_con1 = np.array([c1] * size)
        z_con1 = np.reshape(z_con1, [size, 1])
    else:
        z_con1 = tfd.Uniform(low=-1.0, high=1.0).sample([size, 1])

    if c2 is not None:
        z_con2 = np.array([c2] * size)
        z_con2 = np.reshape(z_con2, [size, 1])
    else:
        z_con2 = tfd.Uniform(low=-1.0, high=1.0).sample([size, 1])

    if cat >= 0:
        z_cat = np.array([cat] * size)
        z_cat = tf.one_hot(z_cat, 10)
    else:
        z_cat = tfd.Categorical(probs=tf.ones([10]) * 0.1).sample([size, ])
        z_cat = tf.one_hot(z_cat, 10)

    noise = tf.concat([z, z_con1, z_con2, z_cat], axis=-1)

    return noise, z_con1, z_con2, z_cat


def train_step(images, step):
    noise, z_con1, z_con2, z_cat = sample(BATCH_SIZE)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        generated_images = generator(noise, training=True)

        real_output, _ = discriminator(images, training=True)
        fake_output, mid = discriminator(generated_images, training=True)
        fqcat, fqcon1, fqcon2 = qnet(mid)

        info_loss, c1, c2, sce = get_mi(fqcon1, fqcon2, fqcat, z_con1, z_con2, z_cat)

        gen_loss = generator_loss(fake_output)
        dis_loss = discriminator_loss(real_output, fake_output)

        gi = gen_loss + info_loss
        di = dis_loss + info_loss

    with writer.as_default():
        tf.summary.scalar("discriminator", dis_loss, step)
        tf.summary.scalar("generator", gen_loss, step)
        tf.summary.scalar("c1", c1, step)
        tf.summary.scalar("c2", c2, step)
        tf.summary.scalar("sce", sce, step)

    gen_grd = gen_tape.gradient(gi, generator.trainable_variables + qnet.trainable_variables)
    dis_grd = dis_tape.gradient(di, discriminator.trainable_variables)

    gen_opt.apply_gradients(zip(gen_grd, generator.trainable_variables))
    dis_opt.apply_gradients(zip(dis_grd, discriminator.trainable_variables))

    return gen_loss, dis_loss


def train(dataset, epochs):
    step = 0
    gen_loss = []
    dis_loss = []
    for epoch in range(epochs):
        for batch in dataset:
            gen, dis = train_step(batch, step)
            writer.flush()
            gen_loss.append(gen)
            dis_loss.append(dis)

            step += 1

            if step % 100 == 0:
                train_display_image(generator, step)

        checkpoint.save(file_prefix=checkpoint_prefix)
        g_loss = tf.reduce_mean(gen_loss).numpy()
        d_loss = tf.reduce_mean(dis_loss).numpy()

        print("{} Generator: {:.4f}\tDiscriminator: {:.4f}".format(epoch + 1, g_loss, d_loss))

    plt.figure(figsize=(20, 8))
    plt.plot(gen_loss, label="generator")
    plt.plot(dis_loss, label="discriminator")
    plt.legend()
    plt.suptitle("GAN loss")
    plt.savefig("GAN loss")


if __name__ == "__main__":
    a = Arguments()
    t = Trainer(a)