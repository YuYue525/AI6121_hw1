import os
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img_dir = "samples"
result_dir = "ahe_results"

def rgb_AHE(img_name, img_path, tileGridSize = (8, 8)):
    img = Image.open(img_path)
    img_array = np.array(img)
    
    img_after_array = np.array(img)
    
    width, height = img.size
    img_size = width * height
    
    tile_height, tile_width = tileGridSize
    tile_size = tile_height * tile_width
    
    # for each pixel
    for h in range(height):
        for w in range(width):
            # for each region
            region_values = []
            for tile_h in range(tile_height):
                for tile_w in range(tile_width):
                
                    #find the pixels of he region
                    pixel_h = abs((h - (tile_height // 2) + tile_h))
                    pixel_w = abs((w - (tile_width // 2) + tile_w))
                    pixel_h = (height - 1 - (pixel_h % (height - 1))) if (pixel_h >= height) else pixel_h
                    pixel_w = (width - 1 - (pixel_w % (width - 1))) if (pixel_w >= width) else pixel_w
                    
                    region_values.append(img_array[pixel_h][pixel_w])
                    
            # for each channel
            img_after_array[h][w][0] = int(round(np.sum(np.array(region_values)[:, 0] <= img_array[h][w][0])/tile_size*255))
            img_after_array[h][w][1] = int(round(np.sum(np.array(region_values)[:, 1] <= img_array[h][w][1])/tile_size*255))
            img_after_array[h][w][2] = int(round(np.sum(np.array(region_values)[:, 2] <= img_array[h][w][2])/tile_size*255))
            # print(img_after_array[h][w][0], img_after_array[h][w][1], img_after_array[h][w][2])
            
    image_after = Image.fromarray(img_after_array)
      
    result_folder_name = "rgb_ahe_results"
    result_folder_path = os.path.join(result_dir, result_folder_name)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    image_after.save(os.path.join(result_dir, result_folder_name, img_name))
             
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
    
def rgb2hsv_AHE(img_name, img_path, tileGridSize = (8, 8)):
    img = Image.open(img_path)
    
    rgb_before_array = np.array(img)
    hsv_before_array = np.array(img, dtype = 'float64')
    rgb_after_array = np.array(img)
    
    width, height = img.size
    img_size = width * height
    
    tile_height, tile_width = tileGridSize
    tile_size = tile_height * tile_width
    
    for x in range(height):
        for y in range(width):
            r, g, b = rgb_before_array[x][y]
            h, s, v = rgb2hsv(r, g, b)
            hsv_before_array[x][y] = np.array([h, s, v], dtype = 'float64')
            
    hsv_after_array = np.array(hsv_before_array, dtype = 'float64')
            
    # for each pixel
    for x in range(height):
        for y in range(width):
            # for each region
            region_values = []
            for tile_h in range(tile_height):
                for tile_w in range(tile_width):
                
                    #find the pixels of the region
                    pixel_h = abs((x - (tile_height // 2) + tile_h))
                    pixel_w = abs((y - (tile_width // 2) + tile_w))
                    pixel_h = (height - 1 - (pixel_h % (height - 1))) if (pixel_h >= height) else pixel_h
                    pixel_w = (width - 1 - (pixel_w % (width - 1))) if (pixel_w >= width) else pixel_w
                    
                    region_values.append(hsv_before_array[pixel_h][pixel_w])
                    
            # for channel
            hsv_after_array[x][y][2] = np.sum(np.array(region_values)[:, 2] <= hsv_before_array[x][y][2])/tile_size
            # print(img_after_array[h][w][0], img_after_array[h][w][1], img_after_array[h][w][2])
            
    for x in range(height):
        for y in range(width):
            h, s, v = hsv_after_array[x][y]
            r, g, b = hsv2rgb(h, s, v)
            rgb_after_array[x][y] = np.array([r, g, b])
            
    image_after = Image.fromarray(rgb_after_array)
      
    result_folder_name = "hsv_ahe_results"
    result_folder_path = os.path.join(result_dir, result_folder_name)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    image_after.save(os.path.join(result_dir, result_folder_name, img_name))
                    
if __name__ == '__main__':
    img_names = os.listdir(img_dir)
    for img_name in img_names:
        if img_name.endswith(".jpg") or img_name.endswith(".jpeg"):
            rgb2hsv_AHE(img_name, os.path.join(img_dir, img_name), tileGridSize = (16, 16))
            
            
    
    
