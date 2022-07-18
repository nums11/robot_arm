from pyfirmata import Arduino, SERVO, util
from time import sleep
import cv2

port = '/dev/ttyACM0'
pin = 9
board = Arduino(port)

board.digital[pin].mode = SERVO

def rotateServo(pin, angle):
  board.digital[pin].write(angle)
  sleep(0.015)

# while True:
# 	for i in range(0, 180):
# 		rotateServo(pin, i)
# 		print(i)

# 	for i in reversed(range(0, 180)):
# 		rotateServo(pin, i)
# 		print(i)

vid = cv2.VideoCapture(2)

for angle in range(181):
  print(angle)
  rotateServo(pin, angle)
  sleep(2)
  ret, frame = vid.read()
  resized = cv2.resize(frame, (256, 256))
  filename = './data/' + str(angle) + '.jpg'
  print(filename)
  cv2.imwrite(filename, resized)

# while True:

# 	# rotateServo(pin, 0)