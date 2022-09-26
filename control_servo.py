from pyfirmata import Arduino, SERVO, util
from time import sleep

port = '/dev/ttyACM0'
pin = 9
board = Arduino(port)
board.digital[pin].mode = SERVO

def rotateServo(angle):
	board.digital[pin].write(angle)
	sleep(0.015)

while True:
	angle = int(input("Servo angle: "))
	rotateServo(angle)