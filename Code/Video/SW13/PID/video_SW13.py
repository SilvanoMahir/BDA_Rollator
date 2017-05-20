#!/usr/bin/env python
#-*- coding: utf-8 -*-

#Defining Python Source Code Encodings
#---------------------------

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from gopigo import *

from PID_CenterDrive import PID
from birdviewThresholdCM import find
from functions import perspective
 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.rotation = 180
camera.resolution = (640, 480)
camera.framerate = 40
rawCapture = PiRGBArray(camera, size=(640, 480))
#minLineLength=200
 
# allow the camera to warmup
time.sleep(0.1)

#Speed of both Motors (0-255)
speed_def = 40
speed_left = speed_def
speed_right =speed_def
#set_speed(speed_def)
turn_r = 0
turn_l = 0
noLineCount = 0
detectLeft = None
detectRight = None
endDetect = 0
n = 0
d = np.zeros((1,1))
#--------------------------------------------------------------------- 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    img = frame.array
    img1 = img
    #cv2.imwrite('orginal.jpg',img1)

    led_off(0)
    led_off(1)

    #Zeitmessung für einen Durchlauf
    if n > 0:
        time_new = int(round(time.time() * 1000))
        print 'time ganze Zeit:' ,(time_new-time_old)
        n = 0
    else:
        time_old = int(round(time.time() * 1000))
        n = 5
    
    #Funktion perspective, erzeugt Vogelansicht
##    e1 = cv2.getTickCount()
    warp = perspective(img1)
    cv2.imwrite('warp.jpg',warp)
##    e2 = cv2.getTickCount()
##    dif = (e2-e1)/cv2.getTickFrequency()
##    print 'Zeit Prespektive:' ,dif


    #gibt zurück ob Ende der Strecke detektiert (Detektiere es 4mal um
    #sicher zu sein das keine Reflexion)
##    found = find(warp)
    found = 0
    if found < 1:
        endDetect = 0
    elif found >= 1 and endDetect > 3 :
        print 'Ende detektiert'
        stop()
        break
    else:
        endDetect += 1
        print 'Mögliches Ende detektiert'
##    print 'Gefundener Sport', found

    #Canny-Filter
    #edges = cv2.Canny(warp,25,75,apertureSize = 3)
    edges = cv2.Canny(warp,50,150,apertureSize = 3)
    #cv2.imwrite('canny.jpg',edges)

    #Hough-Transformation
    minLineLength=10
    lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)

    #Steuert den Motor an 
    if lines is None:       #Wenn keine Linie detektiert 
        stop()
        led_on(0)
        led_on(1)
        if noLineCount > 6:
            stop()
            print 'keine Linie'
            print '________________________________________'
        else:
            #Gedächnis; merkt sich in welcher Richtung unterwegs war
            #versucht den Weg wieder zurück zu finden. Versucht 6mal
            set_speed(speed_def-10)
            fwd()
            d = np.median(d)
            noLineCount += 1
            if d < 0: #rechts korrigieren
                set_right_speed(speed_def-40)
                set_left_speed(speed_def-10)
                time.sleep(0.05)
            elif d >= 0: #links korrigieren
                set_left_speed(speed_def-40)
                set_right_speed(speed_def-15)
                time.sleep(0.05)
            fwd()
            time.sleep(0.01)  
    else:
        noLineCount = 0
        fwd()
        a,b,c = lines.shape
        alpha = range(a)
        d = np.zeros((1,a))
        beta = np.zeros((1,a))
        for i in range(a):
            #Linien Zeichnen herausnehmen
            cv2.line(warp, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 10)
            h = lines[i][0][1]-lines[i][0][3]
            b = lines[i][0][2] - lines[i][0][0]
            l = np.sqrt([(b*b)+(h*h)])
            alpha[i] = (np.arcsin(h/l)*180/np.pi)
            centerX = (lines[i][0][2]-lines[i][0][0])/2 + lines[i][0][0]
            centerY = (lines[i][0][3]-lines[i][0][1])/2 + lines[i][0][1]
            d[0][i]= 380-centerX
            cv2.circle(warp, (centerX,centerY), 15, (0,0,255),-1)
##            cv2.imwrite('houghliness.jpg',warp)
            beta[0][i]= (np.arctan(d[0][i]/centerY)*180/np.pi)
            dError = np.median(beta)
            #print alpha[i]
        angle = int(np.median(alpha))
        
        if abs(angle) >= 85:
            print 'geradeaus'
            if dError > 5:      #Mittellinie links   
                set_right_speed(speed_def)
                set_left_speed(speed_def-15)
                print 'speed right: ', speed_def
                print 'speed left: ', speed_def-15
            elif dError < -5:     #Mittellinie rechts
                set_left_speed(speed_def)
                set_right_speed(speed_def-15)
                print 'speed right: ', speed_def-15
                print 'speed left: ', speed_def
            else:
                set_speed(speed_def)
                print 'speed right: ', speed_def
                print 'speed left: ', speed_def
            speed_right = speed_def
            speed_left = speed_def
            PID(90,beta,d)
            detectLeft = None
            detectRight = None
        elif angle > 0 and abs(angle)< 85:  #drehe rechts
            print 'drehe rechts'
            detectRight = True
            detectLeft = None
            speed_right = int(speed_def-PID(angle,beta,d))
            if speed_right < 20:
                speed_right = 15
            set_right_speed(int(speed_right))
            set_left_speed(speed_def-15)

##            if turn_r > 2:
##                turn_r = 0
##                print 'speed right: ', speed_right
##            else:
##                set_right_speed(speed_def-20)
            print 'speed right: ', speed_right
##                turn_r += 1              
            print 'speed left: ', speed_def-15    
        elif angle < 0 and abs(angle)< 85:  #drehe links
            print 'drehe links'
            detectLeft = True
            detectRight = None
            speed_left = int(speed_left-PID(angle,beta,d))
            if speed_left < 20:
                speed_left = 20
            set_left_speed(int(speed_left))
            set_right_speed(speed_def-20)

            if turn_l > 2:
                turn_l = 0
                print 'speed left: ',speed_left
            else:
                set_left_speed(speed_def-20)
                print 'speed left: ',speed_def-20
                turn_l += 1
            print 'speed right: ', speed_def-20
        print '________________________________________'
    # show the frame
    #warp = cv2.resize(warp,(540,960))
    cv2.imshow("Frame", warp)
    key = cv2.waitKey(1) & 0xFF
 
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
	    break
rawCapture.release()
cv2.destroyAllWindows()
    
