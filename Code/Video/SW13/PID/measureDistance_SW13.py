#!/usr/bin/env python
#-*- coding: utf-8 -*-

#Defining Python Source Code Encodings
#---------------------------

# import the necessary packages
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from gopigo import *
 
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
#--------------------------------------------------------------------- 

set_speed(speed_def)
fwd()
time.sleep(0.7)
stop()
    
