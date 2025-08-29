import cv2 as cv

def plot_cv(img):
    cv.imshow('Image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()