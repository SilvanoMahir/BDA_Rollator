#from controller import PID
import time

Kp = 10
Ki = 0.5
Kd = 0.2
Ti = 1
Td = 1

time_old = int(round(time.time() * 1000))
iValue = 0
lastError= 0
time.sleep(0.5)
dValue, error = 0,0

def PID(angle):
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
