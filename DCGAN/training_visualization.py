import os
import matplotlib.pyplot as plt
# Turn interactive plotting off
plt.ioff()


class Visualization:

    @staticmethod
    def generate_and_save_images(model, epoch, test_input, visualization_folder):
        if not os.path.exists(visualization_folder):
            os.makedirs(visualization_folder)

        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(8, 8))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        result_path = os.path.join(visualization_folder, 'epoch_{:02d}.png')
        plt.savefig(result_path.format(epoch))
