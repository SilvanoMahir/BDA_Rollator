import numpy as np
import cv2
import matplotlib.pyplot as plt

gray = cv2.imread('warp.jpg')
edges = cv2.Canny(gray,100,100,apertureSize = 3)
cv2.imwrite('edges-50-150.jpg',edges)
minLineLength=50
lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)

a,b,c = lines.shape
d = np.zeros((1, a))
beta = np.zeros((1, a))
for i in range(a):
    cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 255, 0), 5, cv2.LINE_AA)
    centerX = (lines[i][0][2]-lines[i][0][0])/2 + lines[i][0][0]
    centerY = (lines[i][0][3]-lines[i][0][1])/2 + lines[i][0][1]
    cv2.circle(gray, (centerX,centerY), 15, (0,0,255),-1)
    d[0][i]= 250-centerX
    cv2.imwrite('Hough_1.jpg',gray)
    beta[0][i]= (np.arctan(d[0][i]/centerY)*180/np.pi)

cv2.line(gray, (250, 0), (250, 600), (255, 0, 0), 3, cv2.LINE_AA)
print d
print beta
plt.figure(figsize=(10,8))
plt.subplot(1,1,1)
plt.title('Zentrum der Linien (Bild Anpassen)')
plt.xlabel('# Pixel')
plt.ylabel('# Pixel')
imgRGB = cv2.cvtColor(gray,cv2.COLOR_BGR2RGB)
fig = plt.imshow(imgRGB)
plt.show()


