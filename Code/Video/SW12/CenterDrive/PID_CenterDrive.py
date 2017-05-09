#from controller import PID
import time
import numpy as np

Kp = 10
Ki = 0.5
Kd = 0.2
Ti = 1
Td = 1
Kp_Center = 1.2
Kp_Center1 = 0.7

detectLeft = None
detectRight = None

time_old = int(round(time.time() * 1000))
iValue = 0
lastError= 0
time.sleep(0.5)
dValue, error = 0,0

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
        
    error = np.median(beta) #abs(0-beta)  = immer beta
    d = np.median(d)

    if d >= 0 and detectLeft == True:      #Mittellinie links und Kurve Links   
        value = value * Kp_Center  
    elif d < 0 and detectRight == True:     #Mittellinie ist rechts und Kurve ist Rechts
        value = value * Kp_Center
    elif d >= 0 and detectRight == True:      #Mittellinie ist links   
        value = value * Kp_Center1  
    elif d < 0 and detectLeft == True:     #Mittellinie ist rechts
        value = value * Kp_Center1
    elif detectLeft == None and detectRight == None and d < 0:  #Mittellinie ist rechts
        value = value * Kp_Center1
    elif detectLeft == None and detectRight == None and d >= 0: #Mittellinie ist links
        

    print 'Value PID:', value
    print 'Beta:', beta
    print 'Abstand zum Mittellinie:', d

    #Saturation 
    if value > 42:
        value = 42           
    elif value < 0:
        value = 0


    

    return value

##while True:
##    value = PID(80)
##    time.sleep(0.5)
##    print(value)
##    print('--------------------------------')
