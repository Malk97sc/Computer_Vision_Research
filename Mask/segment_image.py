import cv2 as cv
import numpy as np

def segment_image(img, threshold = 100): #to use this function the image needs to be in gray scale
    height, width = img.shape
    print(f"Height: {height}, Width: {width}")
    seg_img = np.zeros((height, width))
    area = 0
    for i in range(height):
        for j in range (width):
            if img[i, j] > threshold:
                seg_img[i, j] = 255
                area += 1
    
    print(f"The area of the image in pixels are: {area}")
    return np.uint8(seg_img)