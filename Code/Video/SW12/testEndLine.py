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

camera = PiCamera()

#Funktion perspective in seperatem File
#warp = img1
camera.rotation = 180
camera.resolution = (640, 480)
camera.capture('orginal.jpg')
img1 = cv2.imread('orginal.jpg')
warp = perspective(img1)
cv2.imwrite('warp.jpg',warp)
#cv2.imwrite('warp.jpg',warp)

#gibt zurück ob Ende der strecke detektiert
found = find(warp)
print found
print '________________________________________'

