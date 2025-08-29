import cv2 as cv
import numpy as np

def segment_image(img, threshold = 100):
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    
    height, width = img.shape

    print(f"Height: {height}, Width: {width}")
    seg_img = np.zeros((height, width), dtype = np.uint8)
    area = 0
    for i in range(height):
        for j in range (width):
            if img[i, j] > threshold:
                seg_img[i, j] = 255
                area += 1
    
    print(f"The area of the image in pixels are: {area}")
    return seg_img