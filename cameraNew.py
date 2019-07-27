# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 17:32:46 2019

@author: sbhide
"""
from __future__ import print_function
import cv2 as cv
import numpy as np
import os


win1 = "win1"
win2 = "win2"
hsv= None
frame=None
captureStart1 = 0
captureStart2 = 0
captureStart3 = 0

counter=0
maxfiles = 1500
cap = cv.VideoCapture(0)
#cv.namedWindow(win1)
cv.namedWindow(win2)
cv.namedWindow("win3")
print(os.getcwd())

def create_image(path,imgData,cnt):
    if cnt<maxfiles:        
        print("writing file {0}".format(path))
        cv.imwrite(path,cv.resize(imgData,(50,50)))        
   

def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation = inter)

    # return the resized image
    #print("resized shape is {0}".format(resized.shape))
    return resized

while 1:    
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask1 = cv.inRange(hsv,np.array([0,30,60]),np.array([20,150,255]))
    newmasked = cv.bitwise_and(frame,frame,mask=mask1)
    newmasked= cv.cvtColor(newmasked,cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(newmasked,(5,5),0)
    
    canny = cv.Canny(blurred,0,255)
    kernel = np.ones((2,2),np.uint8)
    opening = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)
    
    #cv.rectangle(opening,(350,10),(610,228),(255,255,255),1)
    ROI = opening[10:228, 350:600]
    
    
    # draw the book contour (in green)
    #cv.rectangle(ROI,(x,y),(x+w,y+h),(255,255,255),5)
    #ROI = ROI[y:y+h,x:x+w]
    #print(contours.shape)
    #ret, thrsh = cv.threshold(ROI,200,255,cv.THRESH_BINARY)
    #cv.imshow(win1, frame)
    cv.imshow(win2, frame)
    cv.imshow("win3",ROI)
    
    if captureStart1==1:
        if not os.path.exists("./data/signature/1"):
            os.mkdir("./data/signature/1")
        
        create_image("./data/signature/1/{0}.png".format(counter),ROI,counter)
        counter+=1
        if counter >maxfiles:
            captureStart1=0
            counter=0
            print("Capturing done for 1")
        
    elif captureStart2==1:
        if not os.path.exists("./data/signature/2"):
            os.mkdir("./data/signature/2")
        
        create_image("./data/signature/2/{0}.png".format(counter),ROI,counter)
        counter+=1
        if counter >maxfiles:
            captureStart2=0
            counter=0
            print("Capturing done for 2")
            
    elif captureStart3==1:
        if not os.path.exists("./data/signature/3"):
            os.mkdir("./data/signature/3")
        
        create_image("./data/signature/3/{0}.png".format(counter),ROI,counter)
        counter+=1
        if counter >maxfiles:
            captureStart3=0
            counter=0
            print("Capturing done for 3")
        

    
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        cap.release()
        cv.destroyAllWindows()
        break
    elif key == ord('1'):
        print("capturing 1 now")
        cv.putText(frame, "Capturing images of 1", (100, 60), cv.FONT_HERSHEY_TRIPLEX, 5, (255, 255, 255))
        captureStart1=1
        
    elif key == ord('2'):
        print("capturing 2 now")
        captureStart2=1
    
    elif key == ord('3'):
        print("capturing 3 now")
        captureStart3=1

        
        
