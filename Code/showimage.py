import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

img1 = cv2.imread("line.jpg", cv2.IMREAD_COLOR)

cv2.imwrite('test.jpg',img1)
