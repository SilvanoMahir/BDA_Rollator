import cv2
from picamera import PiCamera
camera = PiCamera()

camera.rotation = 180
camera.capture('/home/pi/line3.jpg')
