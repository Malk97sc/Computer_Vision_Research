import cv2 
import numpy as np

path = "../Data/hummingbird.jpg"
img = cv2.imread(path)
blur_kernel = (5, 5)
img = cv2.blur(img, blur_kernel)

#output image
height, width, _ = img.shape
output = np.zeros((height, width), np.uint8)

#edges
thres1 = 10
thres2 = 100
canny = cv2.Canny(img, thres1, thres2)

#applying closing to fill edges
clos_kernel = np.ones((3, 3)) # 5x5 or 3x3 is well
closing = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, clos_kernel)

#find contours
contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#draw contours
regions = []
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    regions.append(area)

maxContour = regions.index(max(regions)) #find te biggest
cv2.drawContours(img, contours, maxContour, (0, 255, 0), 3)

#create the output image
cnt = contours[maxContour]
cv2.drawContours(output, [cnt], 0, 255, cv2.FILLED)

cv2.imshow("Binary Image", img)
#cv2.imshow("Canny", closing)
cv2.imshow("Mask", output)
cv2.waitKey(0)
cv2.destroyAllWindows()