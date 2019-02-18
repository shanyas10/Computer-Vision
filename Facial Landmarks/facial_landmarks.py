from imutils import face_utils
import numpy as mp
import argparse
import imutils
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required = True, help = "path to facial landmark predictor")
ap.add_argument("-i", "--image", required = True, help = "path to input image")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector() #This is based on a modification to the standard Histogram of Oriented Gradients + Linear SVM method
predictor = dlib.shape_predictor(args["shape_predictor"])

#load the image and convert to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width = 500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#detect faces
rects = detector(gray, 1) #second parameter is the number of umage pyramid layers to apply when upscaling the image prior to applying detector

#The benefit of increasing the resolution of the input image prior to face detection is that 
# it may allow us to detect more faces in the image — 
# the downside is that the larger the input image, 
# the more computaitonally expensive the detection process is.

#loop over the face detections
for(i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2)

    #show the face number
    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
 
# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)

