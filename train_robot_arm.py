import gym
import os
from TestEnv import TestEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = TestEnv()
log_path = os.path.join(os.getcwd(), 'training_logs')
# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

# model.learn(total_timesteps=25000)

model_save_path = os.path.join(os.getcwd(), 'saved_models', ' PPO_TestEnv2')
# model.save(model_save_path)

model = PPO.load(model_save_path, env=env)
# result = evaluate_policy(model, env, n_eval_episodes=5, return_episode_rewards=True)
# print(result)

episodes = 5
for episode in range(1, episodes+1):
	obs = env.reset()
	score = 0
	steps = 0
	print("Episode", episode)

	while True:
	    action, _states = model.predict(obs)
	    # action = env.action_space.sample()
	    obs, reward, done, info = env.step(action)
	    steps += 1
	    print("Action", action, "state", obs, "reward", reward)
	    score += reward
	    if done: 
	        print('Episode: {} Score: {} Steps: {}\n'.format(episode, score, steps))
	        break