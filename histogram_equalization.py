import os
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img_dir = "samples"
pixel_range = 256
channel_number = 3

def draw_distribution(data):
    plt.hist(data, facecolor='black')
    plt.show()

def rgb_histogram_equalization(img_name, img_path):
    image_before = Image.open(img_path)

    width, height = image_before.size
    image_size = width * height

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

    # draw_distribution(np.array(image_before.getchannel(0)))
    # draw_distribution(np.array(image_after.getchannel(0)))

    image_after.save(os.path.join("results", img_name))

def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    max_rgb = max(r, g, b)
    min_rgb = min(r, g, b)
    m_range = max_rgb - min_rgb

    if max_rgb == min_rgb:
        h = 0
    elif max_rgb == r:
        if g >= b:
            h = 60 * (g-b) / m_range
        else:
            h = 60 * (g - b) / m_range + 360
    elif max_rgb == g:
        h = 60 * (b-r) / m_range + 120
    elif max_rgb == b:
        h = 60 * (r-g) / m_range + 240

    if max_rgb == 0:
        s = 0
    else:
        s = m_range / max_rgb

    v = max_rgb

    return h, s, v

def hsv2rgb(h, s, v):
    h, s, v = float(h), float(s), float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - (f * s))
    t = v * (1 - ((1 - f) * s))

    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q

    r, g, b = int(r * 255), int(g * 255), int(b * 255)

    return r, g, b

def hsv_histogram_equalization(img_name, img_path):
    image_before = Image.open(img_path)

    width, height = image_before.size
    image_size = width * height

    rgb_before_array = np.array(image_before)
    hsv_before_array = np.array(image_before, dtype = 'float64')

    rgb_after_array = np.array(image_before)

    for x in range(height):
        for y in range(width):
            r, g, b = rgb_before_array[x][y]
            h, s, v = rgb2hsv(r, g, b)
            hsv_before_array[x][y] = np.array([h, s, v], dtype = 'float64')

    hsv_after_array = hsv_before_array

    channel_before = hsv_before_array[:, :, 2]

    cdf = []

    for i in range(100000):
        cdf.append(np.sum(channel_before <= (i / 100000))/image_size)

    for x in range(height):
        for y in range(width):
            hsv_after_array[x][y][2] = cdf[int(hsv_before_array[x][y][2]*99999)]

    for x in range(height):
        for y in range(width):
            h, s, v = hsv_after_array[x][y]
            r, g, b = hsv2rgb(h, s, v)
            rgb_after_array[x][y] = np.array([r, g, b])


    image_after = Image.fromarray(rgb_after_array, "RGB")

    # draw_distribution(np.array(image_before.getchannel(0)))
    # draw_distribution(np.array(image_after.getchannel(0)))

    image_after.save(os.path.join("hsv_results", img_name))

if __name__ == '__main__':
    img_names = os.listdir(img_dir)
    for img_name in img_names:
        hsv_histogram_equalization(img_name, os.path.join("samples", img_name))