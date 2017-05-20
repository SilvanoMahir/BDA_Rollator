#!/usr/bin/env python
#-*- coding: utf-8 -*-

#Defining Python Source Code Encodings
#---------------------------

# import the necessary packages
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

array = np.zeros((5,3))
#array  = [gerade, rechts, links]
#       = [......, ....., ...... 
#          ......, ....., ......]
n = 0
old ='kein'

def intell(angle):
    global array, n, old
    angle = int(np.median(angle))
    array[n] = 0
    if abs(angle) >= 85:
        array[n][0] = 1
        old = 'gerade'
        n += 1
    elif angle > 0 and abs(angle) < 85: #drehe rechts
        array[n][1] = 1
        old = 'rechts'
        n += 1
    elif angle < 0 and abs(angle) < 85:
        array[n][2] = 1
        old = 'links'
        n += 1

    if n > 4:
        n = 0  

    summe = sum(array)
    maxi = max(summe)

    if maxi == summe[0] and maxi != summe[1] and maxi != summe[2]:
        old = 'gerade'
        return 'gerade'
    elif maxi == summe[1] and maxi != summe[2] and maxi != summe[0]:
        old = 'rechts'
        return 'rechts'
    elif maxi == summe[2] and maxi != summe[0] and maxi != summe[1]:
        old = 'links'
        return 'links'
    else:
        return old
        


