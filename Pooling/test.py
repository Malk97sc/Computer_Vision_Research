import cv2 
import numpy as np
from pooling_fc import pooling

path = "../Data/coins.jpg"
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

pooled_max = pooling(img, kernel=2, stride=2, mode='max')
pooled_avg = pooling(img, kernel=2, stride=2, mode='avg')

print("Original image shape:", img.shape)
print("Max pooling image shape:", pooled_max.shape)
print("Avg pooling image shape:", pooled_avg.shape)

cv2.imshow("Original", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cv2.imshow("Max Pooling", cv2.cvtColor(pooled_max, cv2.COLOR_RGB2BGR))
cv2.imshow("Avg Pooling", cv2.cvtColor(pooled_avg, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()