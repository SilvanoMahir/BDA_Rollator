import cv2
from picamera import PiCamera
camera = PiCamera()

camera.rotation = 180
camera.resolution = (640, 480)
camera.capture('/home/pi/Desktop/BDA/Take Picture/line5.jpg')
