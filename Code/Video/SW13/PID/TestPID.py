#!/usr/bin/env python
#-*- coding: utf-8 -*-

#Defining Python Source Code Encodings
#---------------------------

# import the necessary packages
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PID_CenterDrive import PID
from birdviewThresholdCM import find

default = 30
inpAngle = 70
beta = 0
d = 0


while True:
    print 'Default speed:', default
    print 'Input angle:', inpAngle
    value = PID(inpAngle,beta,d)
    print 'Set speed:',(default-value)
    print'---------------------'
    time.sleep(0.5)
