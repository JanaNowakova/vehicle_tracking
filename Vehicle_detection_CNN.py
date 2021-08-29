# -*- coding: utf-8 -*-
"""
Created on Thu May 27 23:19:14 2021

@author: Jana Nowakova, Martin Hasal

This program uses OpenCv to detect (and classify) vehicles on video or possible 
on video stream.
Detection is based on YOLOv3 networks on  320x320 and 416x416 resolutions.
Secondary aim is to track vehicles on the road. Since principal goal is to
detect and track vehicles on 2 lines highway by CCTV cameras. Hence the Euclidean
distance is used for tracking. In the future SORT algorithm will be used due to 
varying speed of vehicles, and distance between centers of boundig boxes is not
appropriate metric, it depends on FPS and speed of vehicle.

Codes were motivated by work on these websites:
https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/
https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/

where you can find YOLO structure and weights. As they are not our property.
Or use link few lines after
"""

# imports 
import cv2
import numpy as np
import os # use only needed imports as os.join
import time


# main directory
# set PATH to your root directory
PATH = ''
PATH_IMG = os.path.join(PATH, 'ImageVideo')
# set path to main directory
os.chdir(PATH)

font = cv2.FONT_HERSHEY_PLAIN

# load Yolo weights  # https://pjreddie.com/darknet/yolo/
# YOLOv3 works with different resolution, incease mAP, decrease FPS
cnn_resolution = 416 # possible 416 and 320

if cnn_resolution == 320 :
    cnn = cv2.dnn.readNet("yolov3_320.weights", "yolov3_320.cfg")
if cnn_resolution == 416:
    cnn = cv2.dnn.readNet("yolov3_416.weights", "yolov3_416.cfg")

# write true if run tiny  # tiny needs absolute path otherways it fails
if False:    
    cnn = cv2.dnn.readNet(os.path.join(PATH, "yolov3-tiny.weights"),
                          os.path.join(PATH, "yolov3-tiny.cfg"))
# YOLOv4 tiny    
if False:    
    cnn = cv2.dnn.readNet(os.path.join(PATH, "yolov4-tiny.weights"),
                          os.path.join(PATH, "yolov4-tiny.cfg"))

   
# load clases pretrained by YOLOv3 on COCO dataset
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = cnn.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in cnn.getUnconnectedOutLayers()]
""" be aware of treee output layes from YOLO, can produce different 
classification of one object"""

# !!!!!!!!!! restrict clases !!!!!!!!!!!
# our problem is dedicated to vehicles no cat, dog, horse, etc. are needed
# aim is to speed up program
classes_expected = [0, 1, 2, 3, 5, 7]
# [classes[i] for i in classes_expected]
# GLOBAL CONFIDENCE
CONFIDENCE = 0.5


cap = cv2.VideoCapture( "record_black_lic_plate.avi")
# extrernal camera
#cap = cv2.VideoCapture( 0)
# loop over all images of video 
# create the `VideoWriter()` object
# out_video = cv2.VideoWriter(os.path.join(PATH_IMG,'Result.mp4'), 
#                       cv2.VideoWriter_fourcc(*'mp4v'), 30, 
#                       (int(cap.get(3)), int(cap.get(4))))

nr_frame = 0
while True:
    ret, frame = cap.read()
    
    
    # restrict to region of interest 
    #roi = frame[10: , 340: 1600 ] 
    roi = frame
    height, width, channels = roi.shape
    
    # Object detection
    # original caling 0.00392, change channel to RGB
    blob = cv2.dnn.blobFromImage(roi, 1 / 255.0, (cnn_resolution, cnn_resolution), (0, 0, 0), True, crop=False)
    # start time to calculate FPS
    start = time.time()
    # set the input blob for the convolutional neural network loaded from the disk
    cnn.setInput(blob)
    # forward propagating the blob through the model, which gives us all the outputs
    outs = cnn.forward(output_layers)
    # end time after detection
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end-start)
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    # three outs from YOLO, see articles about, it has different resolutions 
    # for different sizes of objects
    for out in outs:
        # this speed up by taking rows where function confidence > 0.5
        #extract only lines with scores greater than confidence 
        # extract only expected clases
        take = np.take(out[:, 5:],classes_expected, axis=1)
        #out = out[np.where(out[:, 5:] > CONFIDENCE)[0],:]
        #detected_clases = np.argmax(out[:,5:], axis=1)
        detected_clases = np.transpose(np.where(take > CONFIDENCE))
        
        # index to read in oout by rows

        for detection in detected_clases:
            # read the score for every class
            #scores = detection[5:]
            # restrict findig max only for classes in interest
            #class_id = np.argmax(np.take(detection[5:], classes_expected))
            # score of classis with highest rate
            #confidence = scores[classes_expected[class_id]]
            row_out = detection[0]
            confidence = take[row_out,detection[1]]
      
            # Object detected
            center_x = int(out[row_out,[0]] * width)
            center_y = int(out[row_out,[1]] * height)
            w = int(out[row_out,[2]] * width)
            h = int(out[row_out,[3]] * height)
            
            # some of the dected object can be detected variously in 
            # different output layers
            # middle of rectangle, radius 10, thickness 2
            #cv2.circle(roi, (center_x, center_y), radius=5, color=(0, 255, 0), thickness=2)
            #cv2.putText(roi, str(classes[classes_expected[detection[1]]]) + ' ' + str(confidence), 
            #                (center_x, center_y ), font, 1.6, (0, 255, 0), 2)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h]) 
            confidences.append(float(confidence))  
            class_ids.append(detection)
    
            

                
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, 0.1)


    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[classes_expected[class_ids[i][1]]]) + ',' + str(np.round(confidences[i],2))
            color = (0,255,0)
            cv2.rectangle(roi, (x, y), (x + w, y + h), color, 2)
            cv2.putText(roi, label, (x, y ), font, 1.6, color, 2)
            
    
    # put the FPS text on top of the frame
    cv2.putText(roi, f"{fps:.2f} FPS", (20, 30), font, 1, (0, 255, 0), 2) 
    cv2.imshow("Frame", roi)
    # cv2.imwrite("Vehicle.png", roi)
    # 2015 frames video1
    #out_video.write(roi)
    nr_frame +=1
    #print(nr_frame)
    # slow down the speed of video
    key = cv2.waitKey(2)
    if key == 27:
        break







cap.release()
cv2.destroyAllWindows()









# code from to try ssd_mobilenet 
# https://github.com/opencv/opencv/pull/16760
# https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API



"""
# Load a model imported from Tensorflow
tensorflowNet = cv2.dnn.readNetFromTensorflow('ssd_mobilenet_v2_graph.pb', 'ssd_mobilenet_v2.pbtxt')
 
# Input image
img = cv2.imread(os.path.join(PATH_IMG,"car.jpg"))
rows, cols, channels = img.shape
 
# Use the given image as input, which needs to be blob(s).
tensorflowNet.setInput(cv2.dnn.blobFromImage(img, 
                                             scalefactor = (127.5),
                                             size=(300, 300), 
                                             swapRB=True, 
                                             crop=False))
 
# Runs a forward pass to compute the net output
networkOutput = tensorflowNet.forward()
 
# Loop on the outputs
for detection in networkOutput[0,0]:
    
    score = float(detection[2])
    if score > 0.3:
     
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
 
        #draw a red rectangle around detected objects
        cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
 
# Show the image with a rectagle surrounding the detected objects 
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""











