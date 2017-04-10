
import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('Birdview.jpg',0)
plt.subplot(121),plt.imshow(img, cmap='gray')
plt.xlabel('# Pixel')
plt.ylabel('# Pixel')
plt.title('Strecke in Vogelperspektive')
plt.subplot(122),plt.hist(img.ravel(),256,[0,256])
plt.xlabel('Grauwert')
plt.ylabel('Absolute Haeufigkeit')
plt.title('Histogram der Vogelperspektive')
plt.show()
         
