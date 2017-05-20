#from controller import PID
import time
import numpy as np

Kp = 0.0798
Ki = 0.0798
Kd = 0
Ti = 0.236
Td = 1
Kp_Center = 10

detectLeft = None
detectRight = None

time_old = int(round(time.time() * 1000))
iValue = 0
lastError= 0
time.sleep(0.5)
dValue, error = 0,0
speed = np.zeros((1,2))

def PID(angle, beta, d):
    global time_old, iValue, lastError
    time_new = int(round(time.time() * 1000))
    T = time_new - time_old

    error = 90-angle

    #P-Anteil
    pValue = Kp * error
    #I-Anteil
    iValue += Ki/Ti * (error + lastError)/2 * T
    #D-Anteil
    dValue = Kd * Td * (error - lastError) / T
    #print(iValue)


    #Reset I-Value
    if angle > 84:
        iValue  = 0

    #set new values
    lastAngle = angle
    time_old = time_new

    #calculation for output
    value = (pValue + iValue + dValue)/180

    print 'Value PID:', value


    #Mittellinie Anpassungen
    if angle > 0 and abs(angle)< 85:  #Motoren sollen rechts drehen
        detectLeft = None
        detectRight = True
    elif angle < 0 and abs(angle)< 85:  #drehe links
        detectLeft = True
        detectRight = None
    else:
        detectLeft = None
        detectRight = None
        
    error = np.median(beta*np.pi/180) #abs(0-beta)  = immer beta
    print 'error:', error
    d = np.median(d)

    if d >= 0 and detectLeft == True:      #Mittellinie rechts und Kurve links   
        value += Kp_Center * error  
    elif d < 0 and detectRight == True:     #Mittellinie links und Kurve rechts
        value += Kp_Center * error 
    elif d >= 0 and detectRight == True:    #Mittellinie rechts und Kurve rechts  
        value -= Kp_Center * error   
    elif d < 0 and detectLeft == True:     #Mittellinie links und Kurve links
        value -= Kp_Center * error 
    elif detectLeft == None and detectRight == None and d < 0:  #Mittellinie ist rechts
        value += Kp_Center * error 
    elif detectLeft == None and detectRight == None and d >= 0: #Mittellinie ist links
        value += Kp_Center * error

    print 'Value PID + P:', value

    #Saturation 
    if value > 40:
        value = 40           
    elif value < 0:
        value = 10

    print 'Value:', value
##    print 'Beta:', beta
    print 'Abstand d:', d


    

    return value
