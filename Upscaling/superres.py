import cv2
from cv2 import dnn_superres

#https://github.com/Saafke/EDSR_Tensorflow/tree/master/models
 
sr = dnn_superres.DnnSuperResImpl_create()

img = cv2.imread("../Data/Eye.jpg")

path = "Model/EDSR_x3.pb"
sr.readModel(path)

sr.setModel("edsr", 3)

result = sr.upsample(img)

cv2.imwrite("upscaled.png", result)