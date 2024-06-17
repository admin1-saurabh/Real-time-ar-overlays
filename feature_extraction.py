import cv2
import numpy as np



input_img = cv2.imread('saurabh.jpeg')
input_img = cv2.resize(input_img,(400,550),interpolation=cv2.INTER_AREA)
gray_image = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)



# Intiate the orb object

orb = cv2.ORB_create(nfeatures = 1000)

# final keypoints with ORB

keypoints ,descriptors = orb.detectAndCompute(gray_image,None)

#  Drawing the keypoints on the image

final_keypoints = cv2.drawKeypoints(gray_image,keypoints,input_img,(0,255,0))

cv2.imshow("ORB_Keypoints",final_keypoints)
cv2.waitKey()
# print(gray_image)
