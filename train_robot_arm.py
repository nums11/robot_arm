import gym
import os
from RobotArmEnv import RobotArmEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

env = RobotArmEnv()
# log_path = os.path.join(os.getcwd(), 'training_logs')
# model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_path)
# # Stop training when the model reaches the reward threshold
# callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1000, verbose=1)
# eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)
# model.learn(total_timesteps=15000, callback=eval_callback)

model_save_path = os.path.join(os.getcwd(), 'saved_models_v2', ' PPO_RobotArm')
# model.save(model_save_path)

model = PPO.load(model_save_path, env=env)
# result = evaluate_policy(model, env, n_eval_episodes=5, return_episode_rewards=True)
# print(result)

# episodes = 5
# for episode in range(1, episodes+1):
#   obs = env.reset()
#   score = 0
#   steps = 0
#   print("Episode", episode)

#   while True:
#       action, _states = model.predict(obs)
#       # action = env.action_space.sample()
#       obs, reward, done, info = env.step(action)
#       steps += 1
#       print("Taking action", action)
#       # print("Action", action, "state", obs, "reward", reward)
#       score += reward
#       if done: 
#           print('Episode: {} Score: {} Steps: {}\n'.format(episode, score, steps))
#           break

