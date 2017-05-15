#!/usr/bin/env python
#-*- coding: utf-8 -*-

#Defining Python Source Code Encodings
#---------------------------
#Hinzufügen von Packages
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import imutils

##img = cv2.imread("line4.jpg", cv2.IMREAD_COLOR)
##img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def find(img):
        centers = 0
##        plt.figure(figsize=(10,8))
##        fig = plt.imshow(img)

##        plt.subplot(1,2,1)
##        fig = plt.imshow(img)
##        plt.title('Vogelperspektive')

        #Threshold und helle Punkte füllen
        warp = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        warp = cv2.medianBlur(warp,5)
        ret,thresh = cv2.threshold(warp,220,255,cv2.THRESH_BINARY)

        thresh = cv2.erode(thresh, None, iterations=5)
        thresh = cv2.dilate(thresh, None, iterations=10)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        plt.subplot(1,3,3)
        for c in cnts:
                # compute the center of the contour
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                centers += 1
                # draw the contour and center of the shape on the image
##                plt.scatter(cX,cY,zorder=1)

##        plt.title('threshold (v = 220) ')
##        fig = plt.imshow(thresh,cmap ='gray')
##        plt.show()
        
        return centers
