from PID import PID

speed_left = 40

value = PID(90)
speed_left = speed_left - value
print(int(speed_left))
