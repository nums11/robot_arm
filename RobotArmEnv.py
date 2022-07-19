import gym 
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import os
import tensorflow as tf

class RobotArmEnv(Env):
    def __init__(self):
        # Actions we can take, decrease or increase servo angle
        self.action_space = Discrete(180)
        # Image of robot arm
        self.observation_space = Box(low=0.0, high=255.0, shape=(256,256,3), dtype=np.uint8)
        # Set start position
        self.state = self.getImageAsTensor(90)
        # Set episode length
        self.max_episode_length = 50
        self.current_episode_length = 0
        # Zone classification model
        self.zone_classif_model = tf.keras.models.load_model('tf_pose_classifier_v5.h5')
        
    def step(self, action):
        # Write servo position
        if action >= 0 and action <= 180:
            self.state = self.getImageAsTensor(action)
        else:
            print("Error. Invalid action")
            return

        # Increment episode length
        self.current_episode_length += 1
        
        # Calculate reward
        done = False
        zone = self.classifyZone()
        if zone == 'zone 1':
            reward = 1000
            done = True
        elif zone == 'zone 2':
            reward = -1
        elif zone == 'zone 3':
            reward = -10
        else:
            reward = -100

        # Check if max episode length reached
        if self.current_episode_length >= self.max_episode_length: 
            done = True

        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def getImageAsTensor(self, angle):
        zone = self.getZoneFromAngle(angle)
        filename = 'data/zone ' + str(zone) + '/' + str(angle) + '.jpg'
        img_path = os.path.join(os.getcwd(), filename)
        img = tf.keras.utils.load_img(
            img_path, target_size=(256, 256)
        )
        img_array = tf.keras.utils.img_to_array(img)
        return img_array

    def getZoneFromAngle(self, angle):
        if angle <= 20:
            return 1
        elif angle <= 50:
            return 2
        elif angle <= 90:
            return 3
        else:
            return 4

    def classifyZone(self):
        img_array = tf.expand_dims(self.state, 0) # Create a batch
        predictions = self.zone_classif_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        class_names = ['zone 1','zone 2', 'zone 3', 'zone 4']
        zone = class_names[np.argmax(score)]
        return zone

    def render(self):
        pass
    
    def reset(self):
        # Reset state
        self.state = self.getImageAsTensor(90)
        # Reset shower time
        self.current_episode_length = 0
        return self.state