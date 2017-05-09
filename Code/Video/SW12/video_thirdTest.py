# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from gopigo import *
from PID import PID
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
speed_def = 37
speed_left = speed_def
speed_right =speed_def
#set_speed(speed_def)
turn_r = 0
turn_l = 0
noLineCount = 0
detectLeft = None
detectRight = None
n = 0

fwd()
set_speed(40)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    img = frame.array
    img1 = img
    #cv2.imwrite('orginal.jpg',img1)

    print 'go'
    led_off(0)
    led_off(1)

    #Funktion perspective in seperatem File
    #warp = img1
    warp = perspective(img1)
    cv2.imwrite('warp.jpg',warp)

    #gibt zurück ob Ende der strecke detektiert
    found = find(warp)
    if found < 1:
        print 'go'
        fwd()
    else:
        print 'stop'
        stop()
    print found
    print '________________________________________'

    # show the frame
    #warp = cv2.resize(warp,(540,960))
    #cv2.imshow("Frame", warp)
    key = cv2.waitKey(1) & 0xFF
 
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
	    break
rawCapture.release()
cv2.destroyAllWindows()
    
