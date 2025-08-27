import cv2 as cv
import numpy as np

def clean_image(img): 
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    
    height, width = img.shape
    clean = np.zeros((height, width))

    for i in range(height):
        min_value = np.min(img[i, :])
        clean[i, :] = img[i, :] - min_value

    return np.uint8(clean)