from pyfirmata import Arduino, SERVO, util
from time import sleep
import tensorflow as tf
import os
import numpy as np
import cv2
from stable_baselines3 import PPO

# Arduino Setup
port = '/dev/ttyACM0'
pin = 9
board = Arduino(port)

board.digital[pin].mode = SERVO

def rotateServo(angle):
  board.digital[pin].write(angle)
  sleep(0.015)

# TF Setup
img_path = os.getcwd() + 'current_image.jpg'
img_height = 256
img_width = 256

def getCurrentImageAsTensor():
  img = tf.keras.utils.load_img(
    img_path, target_size=(img_height, img_width)
  )
  img_array = tf.keras.utils.img_to_array(img)
  return img_array

# Stable Baselines Setup
rl_model_path = os.path.join(os.getcwd(), 'saved_models_v2', ' PPO_RobotArm')
rl_model = PPO.load(rl_model_path)

# Loop
# define a video capture object
vid = cv2.VideoCapture(0)
while(True):
  # Capture the video frame by frame
  ret, frame = vid.read()
  # Display the resulting frame
  cv2.imshow('frame', frame)
  cv2.imwrite('current_image.jpg', frame)
  state = getCurrentImageAsTensor()
  action, _states = model.predict(state)
  print("Action robot should take", action)
  rotateServo(action)

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
