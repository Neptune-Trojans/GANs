import tensorflow as tf
import numpy as np


class BatchDataGenerator:

    @staticmethod
    def load_data_mnist(buffer_size, batch_size):
        (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
        train_images = train_images[0: buffer_size, :, :, :]
        # Batch and shuffle the data
        train_dataset = tf.data.Dataset. \
            from_tensor_slices(train_images). \
            shuffle(buffer_size, reshuffle_each_iteration=True). \
            batch(batch_size, drop_remainder=True).\
            prefetch(tf.data.experimental.AUTOTUNE)

        return train_dataset

    @staticmethod
    def load_data_cifar(buffer_size, batch_size, label=8):
        """
        airplane : 0
        automobile : 1
        bird : 2
        cat : 3
        deer : 4
        dog : 5
        frog : 6
        horse : 7
        ship : 8
        truck : 9
        :param buffer_size:
        :param batch_size:
        :return:
        """
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
        images = np.concatenate([train_images, test_images])
        labels = np.concatenate([train_labels, test_labels])

        index = np.where(labels == label)
        relevant_images = images[index[0]].astype('float32')
        # relevant_images = relevant_images.reshape(relevant_images.shape[0], 32, 32, 1).astype('float32')
        relevant_images = (relevant_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
        buffer_size = len(relevant_images)
        # Batch and shuffle the data
        train_dataset = tf.data.Dataset. \
            from_tensor_slices(relevant_images). \
            shuffle(buffer_size, reshuffle_each_iteration=True). \
            batch(batch_size, drop_remainder=True).\
            prefetch(tf.data.experimental.AUTOTUNE)

        return train_dataset