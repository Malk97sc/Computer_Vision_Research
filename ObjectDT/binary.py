import cv2 
import numpy as np

path = "../Data/Eye.jpg"
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#binary
thresh = 80
max = 255
_, thre = cv2.threshold(gray, thresh, max, cv2.THRESH_BINARY)

cv2.imshow("Binary Image", thre)
cv2.waitKey(0)
cv2.destroyAllWindows()