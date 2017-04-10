import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Birdview.jpg',0)
#img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
#contours, hierarchy = cv2.findContours(th1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
plt.imshow(img, cmap='gray')
plt.show()

