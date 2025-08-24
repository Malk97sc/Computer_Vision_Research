import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from segment_image import segment_image
from clean_image import clean_image

path = '../Data/rice.jpeg'
img = cv.imread(path, cv.IMREAD_GRAYSCALE)
img.shape

thres = 125
thres_clean = 50

result = segment_image(img, thres)

clean = clean_image(img)
clean_segm = segment_image(clean, thres_clean)

cv.imshow("Without clean", cv.cvtColor(result, cv.COLOR_GRAY2RGB))
cv.imshow("Clean image", cv.cvtColor(clean_segm, cv.COLOR_GRAY2RGB))
cv.waitKey(0)
cv.destroyAllWindows()