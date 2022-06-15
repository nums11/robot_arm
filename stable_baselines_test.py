import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

evironment_name = 'CartPole-v0'
env = gym.make(evironment_name)
log_path = os.path.join(os.getcwd(), 'training_logs')
# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
# I should use a CNN policy
# Observation space will probably be Box(0,255, (256, 256, 3), uint8)
# which means values range from 0 to 255 with 256x256x3 tensor representing image

# model.learn(total_timesteps=20000)

PPO_Path = os.path.join(os.getcwd(), 'saved_models', ' PPO_Model_Cartpole')
# model.save(PPO_Path)

model = PPO.load(PPO_Path, env=env)

# result = evaluate_policy(model, env, n_eval_episodes=1, render=True)
# print(result)

episodes = 5
for episode in range(1, episodes+1):
	obs = env.reset()
	score = 0
	while True:
	    action, _states = model.predict(obs)
	    obs, reward, done, info = env.step(action)
	    score += reward
	    env.render()
	    if done: 
	        print('Episode: {} Score: {}'.format(episode, score))
	        break
env.close()