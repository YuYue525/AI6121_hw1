import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

img_dir = "samples"
img_save_dir = "distribution_plots"

if not os.path.exists(img_save_dir):
    os.makedirs(img_save_dir)

def plot_rgb_distribution(img_name, img_path, save_path):

    img = cv2.imread(img_path)
    _, _, colorChannel = img.shape
    color = ['B', 'G', 'R']

    plt.figure()
    m = []
    for i in range(colorChannel):
        hist_img, _ = np.histogram(img[:, :, i], 256)
        plt.plot(range(256), hist_img, label=color[i])
        plt.legend(loc='best')
        plt.title('histogram_' + img_name)
        m.append(max(hist_img))
    
    m = max(m)
    
    for i in range(colorChannel):
        hist_img, _ = np.histogram(img[:, :, i], 256)
        cdf_img = np.cumsum(hist_img)/img.size * colorChannel * m # accumulative histogram
        plt.plot(range(256), cdf_img, label="scaled_" + color[i]+"_cdf")
        plt.legend(loc='best')

    # print(cdf_img[255])
    plt.savefig(save_path)

def plot_hsv_distribution(img_name, img_path, save_path):

    img = cv2.imread(img_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    plt.figure()
        
    hist_img, _ = np.histogram(img[:, :, 2], 100000)
    plt.plot(range(100000), hist_img, label="luminance")
    plt.legend(loc='best')
    plt.title('histogram_' + img_name)
    
    m = max(hist_img)
    
    hist_img, _ = np.histogram(img[:, :, 2], 100000)
    cdf_img = np.cumsum(hist_img)/img.size * 3 * m # accumulative histogram
    plt.plot(range(100000), cdf_img, label="scaled_luminance_cdf")
    plt.legend(loc='best')

    # print(cdf_img[255])
    plt.savefig(save_path)

if __name__ == '__main__':
    img_names = os.listdir(img_dir)
    for img_name in img_names:
        if img_name.endswith(".jpg") or img_name.endswith(".jpeg"):
            # plot_rgb_distribution(img_name, os.path.join(img_dir, img_name), os.path.join(img_save_dir, 'histogram_' + img_name))
            plot_hsv_distribution(img_name, os.path.join(img_dir, img_name), os.path.join(img_save_dir, 'histogram_' + img_name))
