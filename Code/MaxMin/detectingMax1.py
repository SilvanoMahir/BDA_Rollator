import numpy as np
import cv2
import glob
import argparse
import matplotlib.pyplot as plt

######################################################################
#def hls_color_thresh(img, threshH,threshL, threshS):
def hls_color_thresh(img, threshLow, threshHigh):
    # 1) Convert to HLS color space
    #imgHLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    imgHLS = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #Hue (0,180) Light (0,255), satur (0,255)

   
    # 3) Return a binary image of threshold result
    binary_output = np.zeros((img.shape[0], img.shape[1]))
    #binary_output[(imgHLS[:,:,0] >= threshH[0]) & (imgHLS[:,:,0] <= threshH[1]) & (imgHLS[:,:,1] >= threshL[0]) & (imgHLS[:,:,1] <= threshL[1])  | ((imgHLS[:,:,2] >= threshS[0]) & (imgHLS[:,:,2] <= threshS[1]))] = 1
    binary_output[(imgHLS[:,:,0] >= threshLow[0]) & (imgHLS[:,:,0] <= threshHigh[0]) & (imgHLS[:,:,1] >= threshLow[1])  & (imgHLS[:,:,1] <= threshHigh[1])  & (imgHLS[:,:,2] >= threshLow[2]) & (imgHLS[:,:,2] <= threshHigh[2])] = 1
                 
    return binary_output
##########################################################################

 
image = cv2.imread("line.jpg", cv2.IMREAD_COLOR)
imgRGB = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
imgRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

#Setzen werte Perspektive
src = np.float32([[500,550],[250,1050],[1400,1050],[1250,550]])
dst = np.float32([[0,0],[0,1250],[1000,1250],[1000,0]])

M = cv2.getPerspectiveTransform(src,dst)
Minv = cv2.getPerspectiveTransform(dst,src)

img_size = (imgRGB.shape[1],imgRGB.shape[0])
warp = cv2.warpPerspective(imgRGB.copy(), M, (1000,1250),flags=cv2.INTER_LINEAR) 


# the area of the image with the largest intensity value
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(warp)
cv2.circle(warp, maxLoc, 50, (255, 0, 0), 30)


plt.figure(figsize=(10,8))
plt.subplot(1,3,1)
plt.title('orginal')
fig = plt.imshow(imgRGB)
cv2.line(imgRGB, tuple(src[0]), tuple(src[1]), (255,0,0), 5)
cv2.line(imgRGB, tuple(src[0]), tuple(src[3]), (255,0,0), 5)
cv2.line(imgRGB, tuple(src[3]), tuple(src[2]), (255,0,0), 5)
cv2.line(imgRGB, tuple(src[3]), tuple(src[0]), (255,0,0), 5)
plt.subplot(1,3,3)
fig = plt.imshow(warp, cmap = 'gray')
plt.title('Bird-eye')
plt.subplot(1,3,2)
plt.title('Threshold')
fig = plt.imshow(imgRGB, cmap = 'gray')
plt.show()

cv2.imwrite('test.jpg',warp)
