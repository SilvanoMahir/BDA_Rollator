import numpy as np
import cv2

img = cv2.imread("line.jpg", cv2.IMREAD_COLOR)
imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

scale=3
y_len,x_len,_=img.shape

mean_values=[]
for y in range(scale):
    for x in range(scale):
        cropped_image=img[(y*y_len)/scale:((y+1)*y_len)/scale,
                            (x*x_len)/scale:((x+1)*x_len)/scale]

        mean_val,std_dev=cv2.meanStdDev(cropped_image)
        mean_val=mean_val[:3]

        #(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(cropped_image)
        #maxLoc=mean_val[:3]

        mean_values.append([mean_val])
mean_values=np.asarray(mean_values)
print mean_values.reshape(3,3,3)

