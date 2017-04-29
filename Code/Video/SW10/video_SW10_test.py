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
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(640, 480))
#minLineLength=200
 
# allow the camera to warmup
time.sleep(0.1)

#Speed of both Motors (0-255)
speed_def = 50
speed_left = speed_def
speed_right =speed_def
set_speed(speed_def)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    img = frame.array
    img1 = img

    led_off(0)
    led_off(1)
    
    warp = perspective(img1)
    cv2.imwrite('warp.jpg',warp)

    #Canny-Filter
    #edges = cv2.Canny(warp,25,75,apertureSize = 3)
    edges = cv2.Canny(warp,25,100,apertureSize = 3)

    #Hough-Transformation
    minLineLength=200
    lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)

    #Zeichnet Linien im Bild und berechnet den Winkel von diesen



    #Steuert den Motor an 
    if lines is None:       #Wenn keine Linie detektiert
        stop()
        led_on(0)
        led_on(1)
        print 'keine Linie'
        print '________________________________________'
    else:
        fwd()
        a,b,c = lines.shape
        alpha = range(a)
        for i in range(a):
            cv2.line(warp, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 10)
            cv2.imwrite('houghliness.jpg',warp)
            h = lines[i][0][1]-lines[i][0][3]
            b = lines[i][0][2] - lines[i][0][0]
            l = np.sqrt([(b*b)+(h*h)])
            alpha[i] = (np.arcsin(h/l)*180/np.pi)
            #print alpha[i]
        angle = np.median(alpha)
        if abs(angle) >= 80:
            print 'geradeaus'
            print (angle)
            set_speed(speed_def)
            speed_right = speed_def
            speed_left = speed_def
            PID(90)
            print(speed_def)
        elif angle > 0 and abs(angle)< 80:  #drehe rechts
            print 'drehe rechts'
            speed_right = int(speed_right-PID(abs(angle)))
            if speed_right < 0:
                speed_right = 0
            set_right_speed(speed_right)
            set_left_speed(35)
            print 'speed right: ', speed_right
            print 'speed left: ', speed_left
            print (angle)
        elif angle < 0 and abs(angle)< 80:  #drehe links
            print 'drehe links'
            speed_left = int(speed_left-PID(abs(angle)))
            if speed_left < 0:
                speed_left = 0
            set_left_speed(int(speed_left))
            set_right_speed(35)
            print 'speed right: ', speed_right
            print 'speed left: ',speed_left
            print (angle)

        print '________________________________________'
    # show the frame
    #cv2.imshow("Frame", warp)
    key = cv2.waitKey(1) & 0xFF
 
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
	    break
rawCapture.release()
cv2.destroyAllWindows()
    
