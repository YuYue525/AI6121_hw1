import os
import cv2
import numpy as np

img_dir = "samples"
result_dir = "results_cv2_built_in"

def bgr_histogram_equalization(img_name, img_path):
    img_bgr = cv2.imread(img_path)
    
    img_bgr[:, :, 0] = cv2.equalizeHist(img_bgr[:, :, 0])
    img_bgr[:, :, 1] = cv2.equalizeHist(img_bgr[:, :, 1])
    img_bgr[:, :, 2] = cv2.equalizeHist(img_bgr[:, :, 2])
    
    # create result folder
    result_folder_name = "bgr_results_opencv"
    result_folder_path = os.path.join(result_dir, result_folder_name)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    
    cv2.imwrite(os.path.join(result_folder_path, img_name), img_bgr)

def bgr2yuv_histogram_equalization(img_name, img_path):
    img_bgr = cv2.imread(img_path)
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    # create result folder
    result_folder_name = "bgr2yuv_results_opencv"
    result_folder_path = os.path.join(result_dir, result_folder_name)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    
    cv2.imwrite(os.path.join(result_folder_path, img_name), img_output)
    
def bgr2hsv_histogram_equalization(img_name, img_path):
    img_bgr = cv2.imread(img_path)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
    img_output = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    
    # create result folder
    result_folder_name = "bgr2hsv_results_opencv"
    result_folder_path = os.path.join(result_dir, result_folder_name)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    
    cv2.imwrite(os.path.join(result_folder_path, img_name), img_output)
    
def bgr2lab_histogram_equalization(img_name, img_path):
    img_bgr = cv2.imread(img_path)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    img_lab[:, :, 0] = cv2.equalizeHist(img_lab[:, :, 0])
    img_output = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    
    # create result folder
    result_folder_name = "bgr2lab_results_opencv"
    result_folder_path = os.path.join(result_dir, result_folder_name)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    
    cv2.imwrite(os.path.join(result_folder_path, img_name), img_output)


def bgr_CLAHE(img_name, img_path):
    img_bgr = cv2.imread(img_path)
    
    clahe = cv2.createCLAHE(clipLimit = 10, tileGridSize=(8,8))
    img_bgr[:,:,0] = clahe.apply(img_bgr[:,:,0])
    img_bgr[:,:,1] = clahe.apply(img_bgr[:,:,1])
    img_bgr[:,:,2] = clahe.apply(img_bgr[:,:,2])
    
    result_folder_name = "bgr_CLAHE_results_opencv"
    result_folder_path = os.path.join(result_dir, result_folder_name)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    
    cv2.imwrite(os.path.join(result_folder_path, img_name), img_bgr)

def bgr2yuv_CLAHE(img_name, img_path):
    img_bgr = cv2.imread(img_path)
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    
    clahe = cv2.createCLAHE(clipLimit = 10, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    result_folder_name = "bgr2yuv_CLAHE_results_opencv"
    result_folder_path = os.path.join(result_dir, result_folder_name)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    
    cv2.imwrite(os.path.join(result_folder_path, img_name), img_output)
    
def bgr2hsv_CLAHE(img_name, img_path):
    img_bgr = cv2.imread(img_path)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    clahe = cv2.createCLAHE(clipLimit = 10, tileGridSize=(8,8))
    img_hsv[:,:,2] = clahe.apply(img_hsv[:,:,2])
    img_output = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    
    result_folder_name = "bgr2hsv_CLAHE_results_opencv"
    result_folder_path = os.path.join(result_dir, result_folder_name)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    
    cv2.imwrite(os.path.join(result_folder_path, img_name), img_output)
    
def bgr2lab_CLAHE(img_name, img_path):
    img_bgr = cv2.imread(img_path)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    clahe = cv2.createCLAHE(clipLimit = 10, tileGridSize=(8,8))
    img_lab[:,:,0] = clahe.apply(img_lab[:,:,0])
    img_output = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    
    result_folder_name = "bgr2lab_CLAHE_results_opencv"
    result_folder_path = os.path.join(result_dir, result_folder_name)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    
    cv2.imwrite(os.path.join(result_folder_path, img_name), img_output)
    
if __name__ == '__main__':
    img_names = os.listdir(img_dir)
    for img_name in img_names:
        if img_name.endswith(".jpg") or img_name.endswith(".jpeg"):
            bgr_histogram_equalization(img_name, os.path.join(img_dir, img_name))
            bgr2yuv_histogram_equalization(img_name, os.path.join(img_dir, img_name))
            bgr2hsv_histogram_equalization(img_name, os.path.join(img_dir, img_name))
            bgr2lab_histogram_equalization(img_name, os.path.join(img_dir, img_name))
            bgr_CLAHE(img_name, os.path.join(img_dir, img_name))
            bgr2yuv_CLAHE(img_name, os.path.join(img_dir, img_name))
            bgr2hsv_CLAHE(img_name, os.path.join(img_dir, img_name))
            bgr2lab_CLAHE(img_name, os.path.join(img_dir, img_name))
    
    
