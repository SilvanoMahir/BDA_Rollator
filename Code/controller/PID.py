from controller import PID
import time

Kp = 1
Ki = 0.2
Kd = 0.2

time_old = int(round(time.time() * 1000))
iValue = 0
lastAngle= 0
time.sleep(0.5)
dValue, error = 0,0

def PID(angle):
    global time_old, iValue, lastAngle
    time_new = int(round(time.time() * 1000))
    delta = time_new - time_old

    error = 90-angle

    pValue = Kp * error
    iValue += Ki * error * delta
    dValue = Kd * (angle - lastAngle) / delta
    print(iValue)
    
    lastAngle = angle
    time_old = time_new

    value = (pValue + iValue + dValue)/150

    print(value)

    if value > 60:
        value = 60
    elif value < 0:
        value = 0

    return value

##while True:
##    value = PID(80)
##    time.sleep(0.5)
##    print(value)
##    print('--------------------------------')
