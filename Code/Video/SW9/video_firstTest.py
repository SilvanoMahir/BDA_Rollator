# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from gopigo import *
 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.rotation = 180
camera.resolution = (640, 480)
camera.framerate = 5
rawCapture = PiRGBArray(camera, size=(640, 480))
minLineLength=500
 
# allow the camera to warmup
time.sleep(0.1)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    img = frame.array
    img1 = img
    
    #Setzen werte Perspektive
    src = np.float32([[200,250],[100,470],[550,470],[450,250]])
    dst = np.float32([[0,0],[0,1250],[1000,1250],[1000,0]])
    #Zeichnet Linien in Bild ein, von wo die Vogelperspektive erzuegt wird
    #Achtung: Nimmt Linien auch mit in die warp Variable
##    cv2.line(img1,tuple(src[0]), tuple(src[1]), (255,0,0),1)
##    cv2.line(img1,tuple(src[0]), tuple(src[3]), (255,0,0),1)
##    cv2.line(img1,tuple(src[3]), tuple(src[2]), (255,0,0),1)
##    cv2.line(img1,tuple(src[3]), tuple(src[0]), (255,0,0),1)
    cv2.imwrite('orginal.jpg',img1)

    #Erzeugt die Perspektive
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)

    img_size = (img.shape[1],img.shape[0])
    warp = cv2.warpPerspective(img.copy(), M, (1000,1250),flags=cv2.INTER_LINEAR)
    cv2.imwrite('warp.jpg',warp)

    #Canny-Filter
    edges = cv2.Canny(warp,50,100,apertureSize = 3)

    #Hough-Transformation
    minLineLength=300
    lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)

    #Zeichnet Linien im Bild und berechnet den Winkel von diesen
    #Steuert den Motor an 
    if lines is None:       #Wenn keine Linie detektiert
        right_rot()
        time.sleep(0.05)
        stop()
        print 'keine Linie'
        print '________________________________________'
        
    else:
        a,b,c = lines.shape
        alpha = range(a)
        for i in range(a):
            cv2.line(warp, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 10)
            cv2.imwrite('houghliness.jpg',warp)
            h = lines[i][0][1]-lines[i][0][3]
            b = lines[i][0][2] - lines[i][0][0]
            l = np.sqrt([(b*b)+(h*h)])
            alpha[i] = (np.arcsin(h/l)*180/np.pi)
            print alpha[i]
        angle = np.median(alpha)
        if abs(angle) >= 75:
            print 'geradeaus'
            fwd()
            time.sleep(0.2)
            stop()
            print (angle)
        elif angle > 0 and abs(angle)< 75:  #drehe rechts
            print 'drehe rechts'
            right_rot()
            time.sleep(0.05)
            stop()
            fwd()
            time.sleep(0.2)
            stop()
            print (angle)
        elif angle < 0 and abs(angle)< 75:  #drehe links
            print 'drehe links'
            left_rot()
            time.sleep(0.05)
            stop()
            fwd()
            time.sleep(0.2)
            stop()
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
    
