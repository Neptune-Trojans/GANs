import os
import tensorflow as tf
import shutil
import numpy as np

from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
# Turn interactive plotting off
plt.ioff()


class Visualization:
    def __init__(self, visualization_base_folder, vis_label, vis_cat, vis_noise):
        self.visualization_base_folder = visualization_base_folder
        self._vis_label, self._vis_cat, self._vis_noise = vis_label, vis_cat, vis_noise
        self.fnt = ImageFont.truetype("FreeMonoBold.ttf", 25)

    @property
    def epoch_file_name(self):
        return '{:03d}.png'

    @property
    def gif_file_name(self):
        return 'training_summary.gif'

    @property
    def training_summary_path(self):
        return os.path.join(self.visualization_base_folder, self.gif_file_name)

    @property
    def epoch_file_path(self):
        return os.path.join(self.visualization_base_folder, self.epoch_file_name)

    def save_predicted_images(self, gen_model, q_model, epoch):

        visualization_seed = tf.keras.layers.Concatenate()([self._vis_label, self._vis_cat, self._vis_noise])
        gen_image = gen_model(visualization_seed, training=False)
        cat_output, mu, sigma = q_model(gen_image, training=False)
        mu = mu.numpy().flatten()
        sigma = sigma.numpy().flatten()
        fig = plt.figure(figsize=(18, 18))

        for i in range(gen_image.shape[0]):
            pred_class = np.argmax(cat_output[i])
            pred_class_prob = np.amax(cat_output[i])
            ax = plt.subplot(5, 5, i + 1)
            ax.set_title("class {} ({:.2f}) \n miu {:.2f}, sigma {:.2f}".format(pred_class, pred_class_prob, mu[i], sigma[i]))
            plt.imshow(gen_image[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig(self.epoch_file_path.format(epoch))
        plt.close()

    def generate_gif_image(self):
        frames = []
        for root, dirs, files in os.walk(os.path.abspath(self.visualization_base_folder)):
            for file in sorted(files):
                file_name = os.path.splitext(file)[0]
                img = Image.open(os.path.join(root, file))
                draw = ImageDraw.Draw(img)
                width, height = img.size
                draw.text((width//2, 50), 'epoch {}'.format(file_name), font=self.fnt, fill="black")
                frames.append(img)

        frames[0].save(fp=self.training_summary_path,
                       format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=200,
                       loop=0)

