import os
import numpy as np
from PIL import Image

img_dir = "samples"
pixel_range = 256

def histogram_equalization(img_name, img_path):
    image_before = Image.open(img_path)

    width, height = image_before.size
    image_size = width * height
    channel_number = len(np.array(image_before)[0][0])

    image_after_array = np.array(image_before)
    mapping = []

    for i in range(channel_number):
        channel_before = np.array(image_before.getchannel(i))

        cdf = []

        for j in range(pixel_range):
            cdf.append(np.sum(channel_before <= j)/image_size)

        mapping.append([int(round(255*x)) for x in cdf])

    for x in range(height):
        for y in range(width):
            r = image_after_array[x][y][0]
            g = image_after_array[x][y][1]
            b = image_after_array[x][y][2]
            image_after_array[x][y] = np.array([mapping[0][r], mapping[1][g], mapping[2][b]])

    image_after = Image.fromarray(image_after_array)
    image_after.save(os.path.join("results", img_name))


if __name__ == '__main__':
    img_names = os.listdir(img_dir)
    for img_name in img_names:
        histogram_equalization(img_name, os.path.join("samples", img_name))