import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

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

img1 = cv2.imread("orginal.jpg", cv2.IMREAD_COLOR)
imgRGB = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

#Setzen werte Perspektive
src = np.float32([[100,250],[0,470],[630,470],[550,250]])
dst = np.float32([[0,0],[0,1250],[1000,1250],[1000,0]])

M = cv2.getPerspectiveTransform(src,dst)
Minv = cv2.getPerspectiveTransform(dst,src)

img_size = (imgRGB.shape[1],imgRGB.shape[0])
warp = cv2.warpPerspective(imgRGB.copy(), M, (1000,1250),flags=cv2.INTER_LINEAR)

#Threshold
white_low = np.array([0,0,230])
white_high = np.array([255,50,255])

imgThres_white = hls_color_thresh(warp,white_low,white_high)


plt.figure(figsize=(10,8))
plt.subplot(1,3,1)
plt.title('orginal')
fig = plt.imshow(imgRGB)
plt.scatter(100,250,zorder=1)
plt.scatter(0,470,zorder=1)
plt.scatter(630,470,zorder=1)
plt.scatter(550,250,zorder=1)
##cv2.line(imgRGB,tuple(src[0]), tuple(src[1]), (255,0,0),1)
##cv2.line(imgRGB,tuple(src[0]), tuple(src[3]), (255,0,0),1)
##cv2.line(imgRGB,tuple(src[3]), tuple(src[2]), (255,0,0),1)
##cv2.line(imgRGB,tuple(src[3]), tuple(src[0]), (255,0,0),1)
plt.subplot(1,3,2)
fig = plt.imshow(warp)
plt.title('Bird-eye')
plt.subplot(1,3,3)
plt.title('Threshold')
fig = plt.imshow(imgThres_white,cmap ='gray')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
