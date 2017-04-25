import time

def PID(angle):
    time_new = int(round(time.time() * 1000))
    delta = time_new-time_old

    error = 90-angle

    pValue = Kp * error
    iValue += Ki * error * delta
    dValue = Kd * (angle - lastAngle) / delta

    lastAngle = angle
    time_old = time_new

    value = pValue + iValue + dValue

    if value > 40:
        value = 40
    elif value < 0:
        value = 0

    return value

