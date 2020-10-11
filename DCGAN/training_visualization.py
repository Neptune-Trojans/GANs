import os
import glob
import shutil

from PIL import Image

import matplotlib.pyplot as plt
# Turn interactive plotting off
plt.ioff()


class Visualization:
    def __init__(self, visualization_base_folder, visualization_seed):
        self.visualization_base_folder = visualization_base_folder
        self._generate_vis_folder()
        self._visualization_seed = visualization_seed
        self._generate_vis_folder()

    @property
    def epoch_file_name(self):
        return 'epoch_{:02d}.png'

    @property
    def gif_file_name(self):
        return 'training_summary.gif'

    @property
    def training_summary_path(self):
        return os.path.join(self.visualization_base_folder, self.gif_file_name)

    @property
    def epoch_file_path(self):
        return os.path.join(self.visualization_base_folder, self.epoch_file_name)

    def _generate_vis_folder(self):
        if os.path.exists(self.visualization_base_folder):
            shutil.rmtree(self.visualization_base_folder)
        os.makedirs(self.visualization_base_folder)

    def save_predicted_images(self, model, epoch):

        predictions = model(self._visualization_seed, training=False)

        fig = plt.figure(figsize=(8, 8))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig(self.epoch_file_path.format(epoch))
        plt.close()

    def generate_gif_image(self):
        frames = []
        for root, dirs, files in os.walk(os.path.abspath(self.visualization_base_folder)):
            for file in files:
                frames.append(Image.open(os.path.join(root, file)))

        frames[0].save(fp=self.training_summary_path,
                       format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=200,
                       loop=0)

