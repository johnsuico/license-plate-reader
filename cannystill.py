import cv2 as cv
import numpy as np
import os

originalImg = cv.imread("image.jpg")

if originalImg is None:                             
    print ("error: image not read from file \n\n")        
    os.system("pause")                                          

# Grayscale the image
grayscale = cv.cvtColor(originalImg, cv.COLOR_BGR2GRAY)

# Blur the image using gaussianblur
blurredImg = cv.GaussianBlur(grayscale, (5, 5), 0)

# Get the canny edges of the blurred image
cannyImg = cv.Canny(blurredImg, 100, 200)

cv.namedWindow("Original Image", cv.WINDOW_AUTOSIZE)        
cv.namedWindow("Canny Edges", cv.WINDOW_AUTOSIZE)           

cv.imshow("Original Image", originalImg)         
cv.imshow("Canny Edges", cannyImg)

cv.waitKey()                               

cv.destroyAllWindows()                     