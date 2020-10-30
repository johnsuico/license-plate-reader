import cv2
import numpy as np
import os

def main():
    imgOriginal = cv2.imread("image.jpg")

    if imgOriginal is None:                             
        print ("error: image not read from file \n\n")        
        os.system("pause")                                  
        return                                              
    
    imgGrayscale = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)        # convert to grayscale

    imgBlurred = cv2.GaussianBlur(imgGrayscale, (5, 5), 0)              # blur
    
    imgCanny = cv2.Canny(imgBlurred, 100, 200)                          # get Canny edges

    cv2.namedWindow("imgOriginal", cv2.WINDOW_AUTOSIZE)        
    cv2.namedWindow("imgCanny", cv2.WINDOW_AUTOSIZE)           

    cv2.imshow("imgOriginal", imgOriginal)         
    cv2.imshow("imgCanny", imgCanny)

    cv2.waitKey()                               

    cv2.destroyAllWindows()                     

    return

if __name__ == "__main__":
    main()