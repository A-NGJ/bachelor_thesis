import os

import numpy as np
import torch as T

from td3 import TD3


def polyak_avg(model_params, target_params, tau):
    for phi, target_phi in zip(model_params, target_params):
        target_phi.data.copy_(tau * phi + (1 - tau) * target_phi)

def eval_policy(env, policy: TD3, episodes=10):
    avg_reward = 0
    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
        avg_reward /= episodes
    print('='*20)
    print(f'Average Reward over the Evaluation Step: {avg_reward}')
    print('='*20)
    return avg_reward

def set_seed(seed, env):
    env.seed(seed)
    T.manual_seed(seed)
    np.random.seed(seed)

def get_shapes(env):
    """
    :returns: state shape, action shape, max action
    """
    return env.observation_sapce.shape[0], \
           env.action_space.shape[0], \
           float(env.action_space.high[0])

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
