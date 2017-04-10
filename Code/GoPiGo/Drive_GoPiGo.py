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

angle = np.median(alpha)
if angle <= 100 and angle >= 80:
    print 'geradeaus'
    print (angle)
else:
    if angle <= 90:
        print 'Drehe rechts'
        print (angle)
    else:
        print 'drehe links'
        print (angle)
