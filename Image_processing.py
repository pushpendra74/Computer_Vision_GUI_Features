import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread(r"C:\Users\Pushpendra\Desktop\New folder\Rohit_Kohli.jpg")
W,H,C=img.shape # Numpy pic shape
print(W,H,C)
b,g,r=img[100,50]
print(b,g,r) # Color at 100,50 (x,y) coordinates
img=cv2.resize(img,(600,400))
grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
roi = img[60:160, 320:420] # Cropping and Slicing , Just used as Numpy as array
# Area of Interest/ Region of interest  ,Very Useful while Masking of Image or Videos
center = (W // 2, H // 2) # Tuple for X, Y Coordinates , # We use //  to perform integer math (i.e., no floating point values).
M = cv2.getRotationMatrix2D(center, -45, 1.0) # 45 Clockwise
rotated = cv2.warpAffine(img, M, (W,H))
# Rotation is 3 step process
#  (1) compute the center point (image width and height)
#  (2) a rotation matrix with cv2.getRotationMatrix2D,
#  (3) use the rotation matrix to warp the image with cv2.warpAffine.

Guassianblurred = cv2.GaussianBlur(img, (5, 5), 0) # Kernal filter of 5*5 like used in CNN
medianblur = cv2.medianBlur(img,5) # Take median of Kernal
# Larger kernels would yield a more blurry image. Smaller kernels will create less blurry images
img2 = img.copy()
reactImg=cv2.rectangle(img2, (320, 60), (420, 160), (0, 0, 255), 2)
circleImg=cv2.circle(img2,(W//2,H//2),50,(255,0,0),-1)
lineImg=cv2.line(img2,(W//2,H//2),(W,H),(255,255,0),6)
textImg=cv2.putText(img2,"OpenCV",(75, 25),cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 100), 4)
img3=cv2.imread(r"C:\Users\Pushpendra\Desktop\New folder\Roh_Dhoni.jpg")
img3=cv2.resize(img3,(600,400))
MixImg=cv2.add(img3,img) # For addition Img must be of same size
MixImg2=cv2.addWeighted(img3,0.5,img,0.5,0)
edgedImg = cv2.Canny(img3, 30, 150) # A minimum threshold:30,maxVal : The maximum threshold which is 150
#aperture_size : The Sobel kernel size. By default this value is 3
dilateImg = cv2.dilate(img3,(5,5),iterations=1) # Just smoothen the surface, thicken the boundary , better to follow filters
# Erosions and dilations are typically used to reduce noise in binary images (a side effect of thresholding).
dilateImg5 = cv2.dilate(img3,(5,5),iterations=5) # As Iteration increase the Effect of dilation increase like Epoch in DL
EroedImgCanny=cv2.erode(edgedImg,(5,5),iterations=5)# Reverse of dilation
dilateImg5Canny = cv2.dilate(edgedImg,(5,5),iterations=5)
EroedImgCanny=cv2.erode(edgedImg,(5,5),iterations=5)
EroedImgCannyDilated=cv2.erode(dilateImg5Canny,(5,5),iterations=5)
#Thresholding can help us to remove lighter or darker regions and contours of images.
title=["img","grayImg","img3","rotated","Guassianblurred","medianblur","reactImg","textImg","MixImg","MixImg2","EroedImgCannyDilated","EroedImgCanny"]
image=[img,grayImg,img3,rotated,Guassianblurred,medianblur,reactImg,textImg,MixImg,MixImg2,EroedImgCannyDilated,EroedImgCanny]
for i in range(12): # Simple application of pyplots
    plt.subplot(6,2,i+1) # Start from 0 to 11 as per range
    plt.imshow(image[i])
    plt.title(title[i])
    plt.xticks()
    plt.yticks()
plt.show()





















