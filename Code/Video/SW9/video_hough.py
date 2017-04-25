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
camera.framerate = 20
rawCapture = PiRGBArray(camera, size=(640, 480))
minLineLength=500
 
# allow the camera to warmup
time.sleep(0.1)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        edges = cv2.Canny(image,50,150,apertureSize = 3)

        #Setzen werte Perspektive
        src = np.float32([[500,550],[250,1050],[1400,1050],[1250,550]])
        dst = np.float32([[0,0],[0,1250],[1000,1250],[1000,0]])

        M = cv2.getPerspectiveTransform(src,dst)
        Minv = cv2.getPerspectiveTransform(dst,src)

        img_size = (image.shape[1],image.shape[0])
        warp = cv2.warpPerspective(image.copy(), M, (1000,1250),flags=cv2.INTER_LINEAR)

        lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)
    
        # show the frame
        #cv2.imshow("Frame", warp)
        a,b,c = lines.shape
        alpha = range(a)
        for i in range(a):
            cv2.line(image, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3)
            cv2.imwrite('houghlines5.jpg',image)
            h = lines[i][0][1]-lines[i][0][3]
            b = lines[i][0][2] - lines[i][0][0]
            l = np.sqrt([(b*b)+(h*h)])
            alpha[i] = (np.arcsin(h/l)*180/np.pi)
            print alpha[i]
        key = cv2.waitKey(1) & 0xFF
 
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
 
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                break
rawCapture.release()
cv2.destroyAllWindows()
