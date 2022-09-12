import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2

pixel_range=256
def my_clahe(img,cliplimit,pixel_range):
    if pixel_range>=1 and  cliplimit >=1 :
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
        res=img

    return res

if __name__ == '__main__':
    before_img=cv2.imread("sample01.jpg")
    mb, mg, mr = cv2.split(before_img)

    cliplimit=10000
    my_mb = my_clahe(mb,cliplimit,pixel_range)
    my_mg = my_clahe(mg,cliplimit,pixel_range)
    my_mr = my_clahe(mr,cliplimit,pixel_range)
    res1_img=cv2.merge([my_mb,my_mg,my_mr])
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
    sys_mb = clahe.apply(mb)
    sys_mg = clahe.apply(mg)
    sys_mr = clahe.apply(mr)
    res2_img=cv2.merge([sys_mb,sys_mg,sys_mr])
    cv2.imwrite("res_clahe1.jpg",res1_img)
    cv2.imwrite("res_clahe2.jpg",res2_img)
