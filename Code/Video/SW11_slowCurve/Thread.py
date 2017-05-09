import time
from threading import Thread
import cv2
from functions import perspective
import numpy as np
import matplotlib.pyplot as plt

def myfunc(i):
    print "sleeping 5 sec from thread %d" % i
    time.sleep(5)
    print "finished sleeping from thread %d" % i

def getPerspective1(): 
    e1 = cv2.getTickCount()
    image = cv2.imread("thread.jpg", cv2.IMREAD_COLOR)
    src = np.float32([[200,250],[100,470],[550,470],[450,250]])
    dst = np.float32([[0,0],[0,1250],[1000,1250],[1000,0]])

    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)

    img_size = (image.shape[1],image.shape[0])
    warp = cv2.warpPerspective(image.copy(), M, (1000,1250),flags=cv2.INTER_LINEAR)
    cv2.imwrite('warp.jpg',warp)
    cv2.imshow("Frame",warp)
    e2 = cv2.getTickCount()
    t = (e2 - e1)/cv2.getTickFrequency()
    print t
    return warp

def getPerspective(): 
    e1 = cv2.getTickCount()
    image = cv2.imread("thread.jpg", cv2.IMREAD_COLOR)
    warp = perspective(image)
    cv2.imwrite('warp.jpg',warp)
    cv2.imshow("Frame",warp)
    e2 = cv2.getTickCount()
    t = (e2 - e1)/cv2.getTickFrequency()
    print t
    return warp

for i in range(1):
    t = Thread(target=getPerspective1, args=())
    t.start()

