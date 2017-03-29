import numpy as np
import cv2
import glob
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
img1 = cv2.imread("line.jpg", cv2.IMREAD_COLOR)
imgRGB = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

#Threshold
white_low = np.array([0,0,240])
white_high = np.array([255,50,255])

imgThres_white = hls_color_thresh(imgRGB,white_low,white_high) 

#Setzen werte Perspektive
src = np.float32([[500,550],[250,1050],[1400,1050],[1250,550]])
dst = np.float32([[0,0],[0,1250],[1000,1250],[1000,0]])

M = cv2.getPerspectiveTransform(src,dst)
Minv = cv2.getPerspectiveTransform(dst,src)

img_size = (imgRGB.shape[1],imgThres_white.shape[0])
warp = cv2.warpPerspective(imgThres_white.copy(), M, (1000,1250),flags=cv2.INTER_LINEAR)                                                       
#Beispiel Histogramm
#histogram = np.sum(warp[warp.shape[0]/2:,:], axis=0)
#plt.plot(histogram)

imgGRAY = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# the area of the image with the largest intensity value
blurred = cv2.GaussianBlur(imgGRAY, (11, 11), 0)
thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)

 
# display the results of the naive attempt
cv2.imwrite('tst.png',thresh)


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
fig = plt.imshow(imgThres_white,cmap = 'gray')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
