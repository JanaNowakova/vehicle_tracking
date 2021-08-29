# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 23:05:22 2021

@author: Jana Nowakova, Martin Hasal

This code detects vehicles on record

Features of this version
The roadway is divided into two parts in each part, the vehicle is detected separately.
function img_operations() detects vehicle without shadow

"""

import cv2
import numpy as np
#import matplotlib.pyplot as plt

# help function for image vizualization, uncomment import matplotlib.pyplot
# use plt.imshow(fixColor(img)), replace img to get figure with coordinates
def fixColor(image):
    if len(image.shape) < 3:
        return (cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    elif len(image.shape) == 3:
        return(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# load video
cap = cv2.VideoCapture("outpy2.avi")
# automatic BackgroundSubtractor
subtractor = cv2.createBackgroundSubtractorMOG2(history=300, 
                                                varThreshold=100, 
                                                detectShadows=True)
# get first frame
ret, frame = cap.read()

# FPS 
fps = cap.get(cv2.CAP_PROP_FPS)
totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( "total frames = : ",totalFrames )
videolength = totalFrames/fps
print( "fps = : ",fps )



# smoothing kernels
kernel = np.ones((3,3),np.uint8)
kernel2 = np.ones((15, 15), np.uint8)
kernel3 = np.ones((2,2),np.uint8)


################# lines definitions ################
#######  left line
line_l = np.array( [ [760, 0], [400,1010], [1000,1010], [1000, 0]     ] )
img_l = np.zeros( frame.shape ,np.uint8 ) # create a single channel 200x200 pixel black image 
# draw first line
cv2.fillPoly(img_l, pts =[line_l], color=(0, 255, 0))
# mask of line_1
mask_l = img_l.astype(bool)
# get 2D mask for grayscale
line_l_mask = np.zeros( frame.shape[:2] ,np.uint8 )
# position 1 because  color=(0, 255, 0) other positions are zero
line_l_mask[mask_l[:,:,1]] = 255

#######  right line
line_r = np.array( [ [1020, 0], [1030,1010], [1648,1010], [1260, 0]     ] )
img_r = np.zeros( frame.shape ,np.uint8 ) # create a single channel 200x200 pixel black image 
# draw first line
cv2.fillPoly(img_r, pts =[line_r], color=(255, 0, 0))
# mask of line_r
mask_r = img_r.astype(bool)
# get 2D mask for grayscale
line_r_mask = np.zeros( frame.shape[:2] ,np.uint8 )
# position 1 because  color=(0, 255, 0) other positions are zero
line_r_mask[mask_r[:,:,0]] = 255
# check
# plt.imshow(fixColor(line_r_mask))



# alpha - > oppacity
alpha = 0.5

"""
out = frame.copy()
out[mask_l] = cv2.addWeighted(frame, alpha, img_l, 1 - alpha, 0)[mask_l]
out[mask_r] = cv2.addWeighted(frame, alpha, img_r, 1 - alpha, 0)[mask_r]
cv2.imshow('Output', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


################ functions ###############

def is_vehicle(array, value=255):
    """ function returns total number of white pixels"""
    return np.count_nonzero(array == value)

def img_operations(img, mask, x, y, w, h, plot=False):
    """ process inner frame to cut car without shadow """
    src1_mask=cv2.bitwise_and(img, img, mask=mask)
    img = src1_mask.copy()
    #cv2.imshow("img", img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(gray, 100, 200)
    (cnts, _) = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    # road or shadow? Shadow does not contains big contours
    poss_not_shadow = False
    for cnt in cnts:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(img, [cnt], -1, (255, 0,0 ), 2)
            poss_not_shadow = True
    
    gray = np.float32(gray)
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
    dst = cv2.cornerHarris(gray,2,5,0.04)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None, iterations=1)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.011*dst.max()]=[0,0,255]
    # importovani points from dst to black background
    img_black = np.zeros( gray.shape ,np.uint8 )
    img_black[dst>0.011*dst.max()]= 255
    # smoth isolated points on road
    img_black_morph = cv2.morphologyEx(img_black, cv2.MORPH_OPEN, kernel3, iterations=3)
    x_n, y_n, w_n, h_n = cv2.boundingRect(img_black_morph)
    # turn plot=True to see the process
    if plot:
        cv2.imshow('dst',img)
        cv2.imshow('Corners',img_black)
        cv2.rectangle(img_black_morph, (x_n, y_n), (x_n + w_n, y_n + h_n), 255, 3)
        cv2.imshow('Corners_clusters', img_black_morph)
        print('Original: x, y, w, h' , x, y, w, h, '\n NeW: ', x_n, y_n, w_n, h_n)
    return x_n, y_n, w_n, h_n, poss_not_shadow





while True:
    timer = cv2.getTickCount()
    _, frame = cap.read()

    mask = subtractor.apply(frame)

    ##### object detection
    #cv2.imshow("mask", mask)
    # odstraneni stiny ze MOG2
    _, mask = cv2.threshold(mask, 200, 255,  cv2.THRESH_BINARY)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations=1)
    #cv2.imshow("Opening", opening)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2, iterations=5)
    #_, closing = cv2.threshold(closing, 25, 255,  cv2.THRESH_BINARY)
    #cv2.imshow("Closing", closing)
    
    #### objects in lines
    closing_r = cv2.bitwise_and(closing,line_r_mask)
    #cv2.imshow("closing_r", closing_r)
    closing_l = cv2.bitwise_and(closing,line_l_mask)
    #cv2.imshow("closing_l", closing_l)
    # putting lines together
    #closing = cv2.bitwise_or(closing_r, closing_l)
    
    # check if line contains vehicle, faster than compute canny and other staff
    # first solve left line, value in condition was taken experimentaly by size
    if is_vehicle(closing_l) > 50000:
        canny_l = cv2.Canny(closing_l, 100, 200) # Problem 1 Canny needs boundary close, detects vehicle to late
        (cnts_l, _) = cv2.findContours(canny_l, 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)[-2:]
        for cnt_l in cnts_l:
            # Calculate area and remove small elements
            area_l = cv2.contourArea(cnt_l)
            # contour area sometimes computes small area
            # lenght is addtional check, works better
            lenght_l = cv2.arcLength(cnts_l[0], False)
            if area_l > 50000 or lenght_l > 1000:
                #extract frame 
                x, y, w, h = cv2.boundingRect(cnt_l)
                frame_int_l =  frame[y:y+h,x:x+w]
                #cv2.imshow("frame_int_l", frame_int_l)
                mask_int_l = closing_l[y:y+h,x:x+w]
                #cv2.imshow("mask_int_l", mask_int_l)
                x_n, y_n, w_n, h_n, poss_not_shadow = img_operations(frame_int_l,mask_int_l,x, y, w, h)
                if poss_not_shadow:
                    cv2.rectangle(frame, 
                                  (x+x_n, y+y_n), 
                                  ((x+x_n) + w_n, (y+y_n) + h_n), (0, 255, 0), 3)
                
                
    # check if line contains vehicle, faster than compute canny and other staff
    # first solve left line, value in condition was taken experimentaly by size
    if is_vehicle(closing_r) > 50000:
        
        #print("vehicle")
        canny_r = cv2.Canny(closing_r, 100, 200) 
        (cnts_r, _) = cv2.findContours(canny_r, 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)[-2:]

        for cnt_r in cnts_r:            
            # Calculate area and remove small elements
            area_r = cv2.contourArea(cnt_r)
            lenght_r = cv2.arcLength(cnts_r[0], False)
            if area_r > 50000 or lenght_r > 1000:
                #extract frame 
                x, y, w, h = cv2.boundingRect(cnt_r)
                frame_int_r =  frame[y:y+h,x:x+w]
                #cv2.imshow("frame_int_r", frame_int_r)
                mask_int_r = closing_r[y:y+h,x:x+w]
                #cv2.imshow("mask_int_r", mask_int_r)
                x_n, y_n, w_n, h_n, poss_not_shadow = img_operations(frame_int_r,mask_int_r,x, y, w, h)
                if poss_not_shadow:
                    cv2.rectangle(frame,
                                  (x+x_n, y+y_n), 
                                  ((x+x_n) + w_n, (y+y_n) + h_n), (0, 255, 0), 3)
                
    
        
    
    """
    # if somebody would like to work with original frame
    canny = cv2.Canny(frame, 100, 200)
    (cnts, _) = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cv2.drawContours(frame, cnts, -1, (255, 0, 0), 1)
    for cnt in cnts:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 1000:
            #cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)
            #cv2.drawContours(frame, [cnt], -1, color=(255, 255, 255), thickness=cv2.FILLED)
            x, y, w, h = cv2.boundingRect(cnt)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            ext_frame = frame[y:y+h,x:x+w]
            mask = closing[y:y+h,x:x+w]
            cv2.imshow("Frame_ext", mask)
    """
    # display FPS
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(frame, "FPS : " + str(int(fps)), (300,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
    
    # add coloured masks
    #frame[mask_l] = cv2.addWeighted(frame, alpha, img_l, 1 - alpha, 0)[mask_l]
    #frame[mask_r] = cv2.addWeighted(frame, alpha, img_r, 1 - alpha, 0)[mask_r]
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(2)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
































