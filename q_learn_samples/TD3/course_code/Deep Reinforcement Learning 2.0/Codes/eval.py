import torch
import gym
from gym import wrappers
import numpy as np
from sample_td3 import TD3, mkdir

env_name = "HalfCheetahBulletEnv-v0"
seed = 0
work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')

file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")

eval_episodes = 10
save_env_vid = True
env = gym.make(env_name)
max_episode_steps = env._max_episode_steps
if save_env_vid:
    env = wrappers.Monitor(env, monitor_dir, force = True)
    env.reset()
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
policy = TD3(state_dim, action_dim, max_action)
policy.load(file_name, './pytorch_models/')
avg_reward = 0.0
for _ in range(eval_episodes):
    obs = env.reset()
    done = False
    while not done:
        action = policy.select_action(np.array(obs))
        obs, reward, done, _ = env.step(action)
        env.render()
        avg_reward += reward
avg_reward /= eval_episodes
print ("---------------------------------------")
print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
print ("---------------------------------------")