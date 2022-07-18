import gym 
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np

class TestEnv(Env):
    def __init__(self):
        # Actions we can take, decrease or increase servo angle
        self.action_space = Discrete(2)
        # Servo position
        self.observation_space = Box(low=0, high=180, shape=(1,), dtype=np.uint8)
        # Set start position
        self.state = 90
        # Set episode length
        self.max_episode_length = 300
        self.current_episode_length = 0
        
    def step(self, action):
        # Apply action
        # 0 = decrease servo angle - 1
        # 1 = increase servo angle + 1
        if action == 0:
            self.state -= 1
        elif action == 1:
            self.state += 1
        else:
            print("Error. Invalid action")

        # Increment episode length
        self.current_episode_length += 1
        
        # Calculate reward
        done = False
        if self.state <= 90:
            reward = -1000
        elif self.state > 90 and self.state <= 120:
            reward = -100
        elif self.state > 120 and self.state <= 150:
            reward = -10
        elif self.state > 150 and self.state <= 179:
            reward = -1
        elif self.state == 180:
            reward = 10000
            done = True

        # Check if max episode length reached
        if self.current_episode_length >= self.max_episode_length: 
            done = True

        info = {}
        
        # Return step information
        return [self.state], reward, done, info

    def render(self):
        pass
    
    def reset(self):
        # Reset state
        self.state = 90
        # Reset shower time
        self.current_episode_length = 60 
        return [self.state]