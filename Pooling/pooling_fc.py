import cv2
import numpy as np

def pooling(img, kernel = 2, stride = 2, mode = 'max'):
    height, width, channels = img.shape
    out_he = height // kernel
    out_wi = width // kernel

    pool =  np.zeros((out_he, out_wi, channels), dtype = img.dtype)
    for y in range(0, height - kernel+1, stride):
        for x in range(0, width - kernel+1, stride):
            reg = img[y : y+kernel, x : x+kernel]

            if mode == 'max':
                pool[y // stride, x // stride] = np.max(reg, axis = (0, 1))
            elif mode == 'avg':
                pool[y // stride, x // stride] = np.mean(reg, axis = (0, 1))

    return pool