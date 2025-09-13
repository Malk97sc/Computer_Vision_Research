import matplotlib.pyplot as plt
import cv2 as cv

def plot_cv(img):
    cv.imshow('Image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def plot_img(img, gray = True):
    if gray == False:
        plt.imshow(img)    
    else: 
        plt.imshow(img, cmap = plt.cm.gray)
    plt.axis("off")
    plt.show()