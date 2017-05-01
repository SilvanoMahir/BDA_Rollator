# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from gopigo import *
from PID import PID

from functions import perspective
 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.rotation = 180
camera.resolution = (640, 480)
camera.framerate = 20
rawCapture = PiRGBArray(camera, size=(640, 480))
#minLineLength=200
 
# allow the camera to warmup
time.sleep(0.1)

#Speed of both Motors (0-255)
speed_def = 37
speed_left = speed_def
speed_right =speed_def
set_speed(speed_def)
turn_r = 0
turn_l = 0
noLineCount = 0
detectLeft = None
detectRight = None
n = 0
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    img = frame.array
    img1 = img

    led_off(0)
    led_off(1)

    #Zeitmessung zwischen den beiden Bildern
    if n > 0:
        time_new = int(round(time.time() * 1000))
        print 'time' ,(time_new-time_old)
        n = 0
    else:
        time_old = int(round(time.time() * 1000))
        n = 5

    #Funktion perspective in seperatem File
    warp = perspective(img1)
##    cv2.imwrite('warp.jpg',warp)

    #Canny-Filter
    #edges = cv2.Canny(warp,25,75,apertureSize = 3)
    edges = cv2.Canny(warp,25,100,apertureSize = 3)

    #Hough-Transformation
    minLineLength=150
    lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)

    #Zeichnet Linien im Bild und berechnet den Winkel von diesen



    #Steuert den Motor an 
    if lines is None:       #Wenn keine Linie detektiert
        stop()
        led_on(0)
        led_on(1)
        if noLineCount > 5:
            print 'keine Linie'
            print '________________________________________'
        else:
            set_speed(speed_def-10)
            fwd()
            noLineCount += 1
            if detectRight == True and detectLeft == None:
                set_right_speed(speed_def-40)
                time.sleep(0.05)
            elif detectLeft == True and detectRight == None:
                set_left_speed(speed_def-40)
                time.sleep(0.05)
            fwd()
            time.sleep(0.01)
                 
    else:
        noLineCount = 0
        fwd()
        a,b,c = lines.shape
        alpha = range(a)
        for i in range(a):
            cv2.line(warp, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 10)
##            cv2.imwrite('houghliness.jpg',warp)
            h = lines[i][0][1]-lines[i][0][3]
            b = lines[i][0][2] - lines[i][0][0]
            l = np.sqrt([(b*b)+(h*h)])
            alpha[i] = (np.arcsin(h/l)*180/np.pi)
            #print alpha[i]
        angle = int(np.median(alpha))
        if abs(angle) >= 85:
            print 'geradeaus'
            print (angle)
            set_speed(speed_def)
            speed_right = speed_def
            speed_left = speed_def
            PID(90)
            detectLeft = None
            detectRight = None
            print(speed_def)
        elif angle > 0 and abs(angle)< 85:  #drehe rechts
            print 'drehe rechts'
            detectRight = True
            detectLeft = None
            speed_right = int(speed_right-PID(abs(angle)))
            if speed_right < 0:
                speed_right = 0
            set_right_speed(int(speed_right-10))
            set_left_speed(speed_def-10)

            if turn_r > 1:
                turn_r = 0
            else:
                set_right_speed(speed_def-10)
                turn_r += 1
                 
            print 'speed right: ', speed_right-10
            print 'speed left: ', speed_def-10
            print (angle)      
        elif angle < 0 and abs(angle)< 85:  #drehe links
            print 'drehe links'
            detectLeft = True
            detectRight = None
            speed_left = int(speed_left-PID(abs(angle)))
            if speed_left < 0:
                speed_left = 0
            set_left_speed(int(speed_left)-10)
            set_right_speed(speed_def-10)

            if turn_l > 2:
                turn_l = 0
            else:
                set_left_speed(speed_def-10)
                turn_l += 1
            print 'speed right: ', speed_def-10
            print 'speed left: ',speed_left
            print (angle)
        print '________________________________________'
    # show the frame
    #warp = cv2.resize(warp,(540,960))
##    cv2.imshow("Frame", warp)
    key = cv2.waitKey(1) & 0xFF
 
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
	    break
rawCapture.release()
cv2.destroyAllWindows()
    
