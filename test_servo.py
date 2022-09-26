from pyfirmata import Arduino, SERVO, util
from time import sleep

port = '/dev/ttyACM0'
pin = 9
board = Arduino(port)

board.digital[pin].mode = SERVO

def rotateServo(pin, angle):
	board.digital[pin].write(angle)
	sleep(0.015)

while True:
	for i in range(0, 180):
		rotateServo(pin, i)
		print(i)

	for i in reversed(range(0, 180)):
		rotateServo(pin, i)
		print(i)


# while True:
# 	rotateServo(pin, 90)