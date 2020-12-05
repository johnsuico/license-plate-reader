import numpy as np
import cv2
import pytesseract as tess
import os
import ctypes
from difflib import SequenceMatcher
import pandas as pd

# Global vars
actual = []
predicted = []
accuracy =[]
showSteps = False

def preprocess(img):
  #Preprocessing Function, here we preprocess the imge and retrieve its derivatives, after this step, all the data is numbers we are no longer dealing with the image
  
  #Gaussian Blurring
	imgBlurred = cv2.GaussianBlur(img, (5,5), 0)
    
  #Grayscaling the blurred image
	gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)

	if(showSteps):
		cv2.imshow("Starting Image", img)
		cv2.imshow("Gaussian blur car image", imgBlurred)
		cv2.imshow("Grayscale Car image", gray)

  #Sobel function reduces the overall noise of the image, combinesd Gaussian smoothing and differentiation
  #Detects edges against a darker background
	sobelx = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=3)
    
  #First return is the image which we are ignoring
  #We take the second retrun as ret2, which classifies each pixel as white or black, the black pixels are the numbers to read
  #white will be pixels at the edge of black pixels or irrelavent
	ret2,threshold_img = cv2.threshold(sobelx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	return threshold_img

def cleanPlate(plate):
	gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
	_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
	contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	if contours:
		areas = [cv2.contourArea(c) for c in contours]
		max_index = np.argmax(areas)

		max_cnt = contours[max_index]
		max_cntArea = areas[max_index]
		x,y,w,h = cv2.boundingRect(max_cnt)

		if not check_ratio(max_cntArea,w,h):
			return plate,None

		cleaned_final = thresh[y:y+h, x:x+w]
		return cleaned_final,[x,y,w,h]

	else:
		return plate,None

def extract_contours(threshold_img):
    
    #Extracts the contours from the threshold image we created earlier
    #Retreives the shape of pixels we want, in this case we are looking for a rectangle
	element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
	morph_img_threshold = threshold_img.copy()
    
    #MorphologyEx function removes excess noise from a shape such as whitespace opeings and fly pixels
	cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
    
    #finds the contours of the rectangles we narrowed down to
	contours, hierarchy= cv2.findContours(morph_img_threshold,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
	return contours

def check_ratio(area, width, height):
	ratio = float(width) / float(height)
	if ratio < 1:
		ratio = 1 / ratio

	aspect = 4.7272
	min = 15*aspect*15  # minimum area
	max = 125*aspect*125  # maximum area

	rmin = 2
	rmax = 7

	if (area < min or area > max) or (ratio < rmin or ratio > rmax):
		return False
	return True

# Calculate white level
def whiteness(plate):
	avg = np.mean(plate)
	if(avg>=30):
		return True
	else:
 		print('Plate is not white enough, cannot be read')
 		return False

def validateRotationAndRatio(rect):
  # Extract attributes of rectangle
	(x, y), (width, height), rect_angle = rect

  # Check width and height
	if(width>height):
		angle = -rect_angle
	else:
		angle = 90 + rect_angle

  # Check angle
	if angle>15:
	 	print ("Plate angle is too skewed, could not be read")
	 	return False

  # Check width and height if 0
	if height == 0 or width == 0:
		print('area = 0')
		return False

  #Calculate area and see if it is the proper ratio
	area = height*width
	if not check_ratio(area,width,height):
		print('Proper ratio not found')
		return False
	else:
		return True

def readLicense(img, filename):
  # Get original image
  originalImg = img

  # Check if originalImg is not empty
  if originalImg is None:                             
      print ("error: image not read from file \n\n")        
      os.system("pause")                                          

  actual_license_plate = []       # List to store actual license plate string
  predicted_license_plate = []    # List to store predicted license plate string

  actual_license_plate.append(filename[:-4])  # Take filename and remove .jpg or .png

  # Predicted result using pytesseract
  predicted_result = tess.image_to_string(originalImg, lang = 'eng', config="-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6 --oem 3")

  # Replace all : and - with white spaces
  filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
  predicted_license_plate.append(filter_predicted_result) # Add predicted results into list

  # Resize license plate by a factor of 2
  resize_test_license_plate = cv2.resize(originalImg, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)

  # Grayscale the cropped resized image
  grayscale_resize_test_license_plate = cv2.cvtColor(resize_test_license_plate, cv2.COLOR_BGR2GRAY)

  # Blur the image
  gaussian_blur_license_plate = cv2.GaussianBlur(grayscale_resize_test_license_plate, (5, 5), 0)

  # Predict the license plate
  predicted_result = tess.image_to_string(gaussian_blur_license_plate, lang = 'eng', config="-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6 --oem 3")

  # Replace all : and - with white spaces
  filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
  predicted_license_plate.append(filter_predicted_result) 

  # Only include predicted license plates that are at least above half the length of actual
  if ((len(actual_license_plate[0])/2) <= len(predicted_license_plate[0])):
    predicted.append(predicted_license_plate[0])

  # Show image processing steps if chosen
  if (showSteps):
    cv2.imshow("Cropped and resize license plate", resize_test_license_plate)
    cv2.imshow("Grayscale image", grayscale_resize_test_license_plate)
    cv2.imshow("Gaussian blur plate", gaussian_blur_license_plate)
    ctypes.windll.user32.MessageBoxW(0, predicted_license_plate[0], "Predicted License Plate Text", 1)

def cleanAndRead(img, contours, filename):
  for i, cnt in enumerate(contours):
    min_rect = cv2.minAreaRect(cnt) # Draw a rectangle

    # Validate the ratio
    if validateRotationAndRatio(min_rect):
      x, y, w, h = cv2.boundingRect(cnt)
      plate_img = img[y:y+h, x:x+w]

      # Check white levels
      if(whiteness(plate_img)):
        clean_plate, rect = cleanPlate(plate_img)
        
        # If there is a rectangle
        if rect:
          x1, y1, w1, h1 = rect
          x, y, w, h = x+x1, y+y1, w1, h1
          img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw rectangle using green outline on the image
          crop_img = img[y:y+h, x:x+w]    # Crop image based on rectangle points
          if (showSteps):   # Show steps if chosen
            cv2.imshow("Detected Plate", img)
          readLicense(crop_img, filename)   # Read characters of license plate and pass through cropped image and filename

# Calculate similarity accuracy
def similar(a, b):
  return SequenceMatcher(None, a, b).ratio()

def calcAccuracy(diffAcc):
 # Go through each actual and predicted license plate and calculate the accuray
  for i in range(len(actual)):

    if (not diffAcc):
      # ACCURACY BY STRING SIMILARITY
      # ===========================================================
      sim = str(round(similar(actual[i], predicted[i])*100, 2))
      accuracy.append(sim)
      # ===========================================================

    else:
      # ACCURACY USING CHAR BY CHAR COMPARISON
      # ===========================================================
      correct = 0
      if (len(actual[i]) >= len(predicted[i])):
        maxLength = len(predicted[i])
        for x in range(len(predicted[i])):
          if(actual[i][x] == predicted[i][x]):
            correct += 1
      else:
        maxLength = len(actual[i])
        for x in range(len(actual[i])):
          if(actual[i][x] == predicted[i][x]):
            correct += 1

      acc = str(round(((correct/maxLength)*100), 2))
      accuracy.append(acc)
      # ===========================================================

  # Putting all the data in a dataframe to be displayed
  print('\n\n')
  if (diffAcc):
    print('Comparison Method')
  else:
    print('Similarity method')
  data = {'Actual': actual,
          'Predicted': predicted,
          'Accuracy': accuracy}
  df = pd.DataFrame(data=data)
  print(df)
  actual.clear()
  predicted.clear()
  accuracy.clear()

# Plate detection
print ("Plate Detection in Progress...")

# Setting up the path for testing
dir_path = os.path.dirname(os.path.realpath(__file__))
license_path = dir_path + '/License-Plates'

# Menu choice var
choice = 0

while choice == 0:
  print('\n\n\n============================================================')
  print('Please choose an option below to use to continue')
  print('1. Show steps with a sample image')
  print('2. Go through all the test images and show accuracies (similarity method)')
  print('3. Go through all the test images and show accuracies (comparison method)')
  print('4. Show steps with a not working image')
  print('5. Exit application')
  print('============================================================')
  choice = input("Please enter a number: ")

  if (choice == '1'):
    showSteps=True                                                      # Set show steps to true
    img = cv2.imread('HH9999MM.jpg')                                    # Read in best sample image
    # img = cv2.imread('10.jpg')                                           # Read in not working sample image   
    threshold_img = preprocess(img)                                     # Get threshold image
    contours = extract_contours(threshold_img)                          # Get contours
    cleanAndRead(img, contours, '1.jpg')                                # Clean the image and read the plate
    choice = 0                                                          # Go back to main menu

  if (choice == '2'):
    # Go through all the files in License-Plates folder
    showSteps=False
    diffAcc = False
    for filename in os.listdir(license_path):
      if filename.endswith(".jpg") or filename.endswith(".png"):
        print('Detecting plates for file: ' + filename)                 # Just to give the user a status report while detecting
        actual.append(filename[:-4])                                    # Get the file name and remove the .png or .jpg extension
        img = cv2.imread(license_path + '/' + filename)                 # Read in the image from folder
        threshold_img = preprocess(img)                                 # Get threshold image
        contours = extract_contours(threshold_img)                      # Get contours
        cleanAndRead(img, contours, filename)                           # Clean the image and detect what the plate says
    calcAccuracy(diffAcc)
    choice = 0

  if (choice == '3'):
    # Go through all the files in License-Plates folder
    showSteps=False
    diffAcc = True
    for filename in os.listdir(license_path):
      if filename.endswith(".jpg") or filename.endswith(".png"):
        print('Detecting plates for file: ' + filename)                 # Just to give the user a status report while detecting
        actual.append(filename[:-4])                                    # Get the file name and remove the .png or .jpg extension
        img = cv2.imread(license_path + '/' + filename)                 # Read in the image from folder
        threshold_img = preprocess(img)                                 # Get threshold image
        contours = extract_contours(threshold_img)                      # Get contours
        cleanAndRead(img, contours, filename)                           # Clean the image and detect what the plate says
    calcAccuracy(diffAcc)
    choice = 0

  if (choice == '4'):
    showSteps=True                                                      # Set show steps to true
    img = cv2.imread('TN01AS9299.jpg')                                           # Read in not working sample image   
    threshold_img = preprocess(img)                                     # Get threshold image
    contours = extract_contours(threshold_img)                          # Get contours
    cleanAndRead(img, contours, 'TN01AS9299.jpg')                                # Clean the image and read the plate
    choice = 0   

  if (choice == '5'):
    break