import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('warp.jpg',0)
plt.subplot(121),plt.imshow(img, cmap='gray')
plt.subplot(122),plt.hist(img.ravel(),256,[0,256])
plt.show()
