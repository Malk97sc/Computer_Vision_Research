import cv2 as cv
import numpy as np

def clean_image(img): #to use this function the image needs to be in gray scale
    height, width = img.shape
    clean = np.zeros((height, width))

    for i in range(height):
        min_value = np.min(img[i, :])
        clean[i, :] = img[i, :] - min_value

    return np.uint8(clean)