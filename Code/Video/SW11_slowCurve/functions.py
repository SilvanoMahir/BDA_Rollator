import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from gopigo import *

def perspective(image):
    #Setzen werte Perspektive
    src = np.float32([[100,200],[0,470],[630,470],[550,200]])
    dst = np.float32([[0,0],[0,1250],[1000,1250],[1000,0]])
    #Zeichnet Linien in Bild ein, von wo die Vogelperspektive erzuegt wird
    #Achtung: Nimmt Linien auch mit in die warp Variable
##    cv2.line(image,tuple(src[0]), tuple(src[1]), (255,0,0),1)
##    cv2.line(image,tuple(src[0]), tuple(src[3]), (255,0,0),1)
##    cv2.line(image,tuple(src[3]), tuple(src[2]), (255,0,0),1)
##    cv2.line(image,tuple(src[3]), tuple(src[0]), (255,0,0),1)
    cv2.imwrite('orginal.jpg',image)

    #Erzeugt die Perspektive
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)

    img_size = (image.shape[1],image.shape[0])
    warp = cv2.warpPerspective(image.copy(), M, (1000,1250),flags=cv2.INTER_LINEAR)
    return warp
