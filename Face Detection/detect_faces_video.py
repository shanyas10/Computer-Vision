import numpy as np
import argparse
import cv2
import imutils
from imutils.video import VideoStream
import time

#Constructing argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required = True, help = "path to Caffe deploy prototxt file")
ap.add_argument("-m", "--model", required = True, help = "path to Caffe pretrained model")
ap.add_argument("-c", "--confidence", type = float, default = 0.5, help = "minimum probability to filter weak detections")
args = vars(ap.parse_args())

#load serialized model form disk
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#initialize Video Stream
vs = VideoStream(src=0).start()
time.sleep(2.0)

#loop over the frames

while True:
    frame = vs.read()
    frame = imutils.resize(frame,width = 400)

    #convert frame to blob
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300,300),(104.0, 177.0, 123.0))

    #pass the blob through network and obtain detections
    net.setInput(blob)
    detections = net.forward()

    #loop over the detections 
    for i in range (0, detections.shape[2]):
        confidence = detections[0,0,i,2]

        #extract confidence
        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            #compute coordinates
            text = "{:.2f}%".format(confidence * 100)
            y = startY-10 if startY-10>10 else startY+10
            
            #bound rectangle
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    # show the output image
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF

    #condition for quitting
    if key == ord("q"):
        break
#Cleanup
cv2.destroyAllWindows()
vs.stop()