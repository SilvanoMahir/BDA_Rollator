# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import numpy as np
import cv2
 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.rotation = 180
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.1)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        img = frame.array
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        #Setzen werte Perspektive
        src = np.float32([[200,250],[100,470],[550,470],[450,250]])
        dst = np.float32([[0,0],[0,1250],[1000,1250],[1000,0]])

        M = cv2.getPerspectiveTransform(src,dst)
        Minv = cv2.getPerspectiveTransform(dst,src)

        img_size = (img.shape[1],img.shape[0])
        warp = cv2.warpPerspective(img.copy(), M, (1000,1250),flags=cv2.INTER_LINEAR)
    
        image = cv2.Canny(warp,100,100,apertureSize = 3)
 
        # show the frame
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF
 
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
 
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                break
rawCapture.release()
cv2.destroyAllWindows()
