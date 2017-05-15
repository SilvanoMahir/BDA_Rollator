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
 
#Speed of both Motors (0-255)
speed_def = 40
speed_left = speed_def
speed_right =speed_def

#--------------------------------------------------------------------- 

set_speed(speed_def)
fwd()
time.sleep(0.7)
stop()
    
