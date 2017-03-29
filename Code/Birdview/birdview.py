import numpy as np
import cv2 
import glob
import matplotlib.pyplot as plt

img1 = cv2.imread("line.jpg", cv2.IMREAD_COLOR)
src = np.float32([[500,550],[250,1050],[1400,1050],[1250,550]])
#src = np.float32([[1250,1050],[1250,550],[250,550],[250,1050]])
dst = np.float32([[100,0],[0,720],[500,720],[500,0]])
#dst = np.float32([[500,720],[500,0],[0,0],[0,720]])

M = cv2.getPerspectiveTransform(src,dst)
Minv = cv2.getPerspectiveTransform(dst,src)

img_size = (img1.shape[1],img1.shape[0])
warp = cv2.warpPerspective(img1.copy(), M, img_size)
#cv2.imshow('transform',warp)

plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
plt.title('orginal')
fig = plt.imshow(img1)
plt.scatter(500,550,zorder=1)
plt.scatter(250,1050,zorder=1)
plt.scatter(1400,1050,zorder=1)
plt.scatter(1250,550,zorder=1)
plt.subplot(2,1,2)
fig = plt.imshow(warp)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
