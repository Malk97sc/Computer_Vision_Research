import cv2 

path_img = "../Data/Perseus.jpg"
img = cv2.imread(path_img)

#Bin image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = 150
maxval = 255
_, thres = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#Find contours
contour, _ = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contour, -1, (0, 255, 0), 2) 

#show
#img_resized = cv2.resize(img, (640, 480)) #to show better
cv2.imshow("Contours", img)
cv2.waitKey(0)
cv2.destroyAllWindows()