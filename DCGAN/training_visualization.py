import matplotlib.pyplot as plt
# Turn interactive plotting off
plt.ioff()


class Visualization:

    @staticmethod
    def generate_and_save_images(model, epoch, test_input):

        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(8, 8))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig('epoch_{:04d}.png'.format(epoch))
