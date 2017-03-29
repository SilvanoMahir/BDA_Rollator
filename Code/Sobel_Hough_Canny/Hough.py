import numpy as np
import cv2
import matplotlib.pyplot as plt

gray = cv2.imread('Birdview.jpg')
gray = cv2.cvtColor(gray,cv2.COLOR_BGR2RGB)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
cv2.imwrite('edges-50-150.jpg',edges)
minLineLength=500
lines = cv2.HoughLinesP(image=edges,rho=5,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)

a,b,c = lines.shape
alpha = range(a)
for i in range(a):
    cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3)
    cv2.imwrite('houghlines5.jpg',gray)
    h = lines[i][0][1]-lines[i][0][3]
    b = lines[i][0][2] - lines[i][0][0]
    l = np.sqrt([(b*b)+(h*h)])
    alpha[i] = (np.arcsin(h/l)*180/np.pi)
    print alpha[i]

    #cv2.line(gray,(0,0),(643,402),(255, 0, 0), 3, cv2.LINE_AA)
    
    
plt.figure(figsize=(10,8))
#plt.subplot(1,1,1)
plt.title('Houghtransformation in Vogel-Perspektive')
fig = plt.imshow(gray)
plt.show()
