import tensorflow as tf


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
