import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2

pixel_range = 256
img_dir = "samples"
result_dir = "clhe_results"

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

def my_clhe(img, cliplimit, pixel_range):
    if pixel_range>=1 and cliplimit >=1 :
        height, width = img.shape
        s = np.zeros((pixel_range,), dtype=int)

        for i in range(height): #calculate original histgram
            for j in range(width):
                s[img[i, j]] = s[img[i, j]] + 1

        clipped = 0 # use cliplimit to redistribute the histgram
        for i in range(pixel_range):
            if s[i] > cliplimit:
                clipped = clipped + s[i] - cliplimit
                s[i] = cliplimit
        resdistBatch = int(clipped / pixel_range)
        residual = clipped - resdistBatch * pixel_range
        for i in range(pixel_range): # average distribution
            s[i] = s[i] + resdistBatch
        for i in range(residual):
            s[i] = s[i] + 1


        sum = np.zeros((pixel_range,), dtype=int) #calculate cumulative matrix
        for i in range(pixel_range):
            if i == 0:
                sum[i] = s[i]
            else:
                sum[i] = sum[i - 1] + s[i]
        for i in range(pixel_range):
            sum[i] = sum[i] / (height * width) * pixel_range

        res = np.zeros((height, width), dtype=int)
        for i in range(height):
            for j in range(width):
                res[i, j] = sum[img[i, j]]
    else:
        res = img

    return res

if __name__ == '__main__':
    img_names = os.listdir(img_dir)
    for img_name in img_names:
        if img_name.endswith(".jpg") or img_name.endswith(".jpeg"):
        
            #RGB
            RGB_folder_name = "rgb_clhe_results"
            if not os.path.exists(os.path.join(result_dir, RGB_folder_name)):
                os.makedirs(os.path.join(result_dir, RGB_folder_name))
            
            before_img = cv2.imread(os.path.join(img_dir, img_name))
            mb, mg, mr = cv2.split(before_img)

            cliplimit = 10000
            
            my_mb = my_clhe(mb,cliplimit,pixel_range)
            my_mg = my_clhe(mg,cliplimit,pixel_range)
            my_mr = my_clhe(mr,cliplimit,pixel_range)
            res1_img = cv2.merge([my_mb,my_mg,my_mr])

            cv2.imwrite(os.path.join(result_dir, RGB_folder_name, img_name), res1_img)
            
            #HSV
            
            HSV_folder_name = "hsv_clhe_results"
            if not os.path.exists(os.path.join(result_dir, HSV_folder_name)):
                os.makedirs(os.path.join(result_dir, HSV_folder_name))
            
            before_img = cv2.imread(os.path.join(img_dir, img_name))
            before_img = cv2.cvtColor(before_img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(before_img)
            

            cliplimit = 10000
            v_ = my_clhe(v, cliplimit, pixel_range)
            
            for x in np.nditer(v_,  op_flags=['readwrite']):
                if x<0:
                    x[...] = 0
                elif x > 255:
                    x[...] = 255
        

            res1_img = cv2.merge([h, s, np.array(v_, dtype = 'uint8')])
            res1_img = cv2.cvtColor(res1_img, cv2.COLOR_HSV2BGR)

            cv2.imwrite(os.path.join(result_dir, HSV_folder_name, img_name), res1_img)
            
            #opencv clahe
            '''
            clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
            sys_mb = clahe.apply(mb)
            sys_mg = clahe.apply(mg)
            sys_mr = clahe.apply(mr)
            res2_img=cv2.merge([sys_mb,sys_mg,sys_mr])
            
            cv2.imwrite("res_clahe2.jpg",res2_img)
            
            '''
