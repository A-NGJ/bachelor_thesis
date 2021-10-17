import argparse
import os
import time

import gym
from gym import wrappers

from td3 import TD3
import util


def main(**kwargs):
    filename = f'TD3_{kwargs["env"]}_{int(kwargs["seed"])}'

    print('-'*20)
    print(f'Settings: {filename}')
    print('-'*20)

    if kwargs["save"] and not os.path.exists('./pytorch_models'):
        os.mkdir('./pytorch_models')

    env = gym.make(kwargs["env"])
    util.set_seed(kwargs['seed'], env)
    state_shape, action_shape, max_action = util.get_shapes(env)

    policy = TD3(state_shape, action_shape, max_action)

    evalutaions = [util.eval_policy(env, policy)]

    workdir = util.mkdir('exp', 'brs')
    monitor_dir = util.mkdir(workdir, 'monitor')

    max_episode_steps = env._max_episode_steps
    save_env_vid = False
    if save_env_vid:
        env = wrappers.Monitor(env, monitor_dir, force=True)
        env.reset()

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_n = 0
    done = True
    t0 = time.time()

    # TODO: Training loop

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='Environment name',
                        default='HalfCheetahBulletEnv-v0', type=str)
    parser.add_argument('--seed', default=0, type=str)
    parser.add_argument('--start-timesteps', default=1e4, type=int)
    parser.add_argument('--eval_freq', help='Frequency of the evaluation step',
                        default=5e3, type=int)
    parser.add_argument('--max-timesteps', help='Total number of timesteps',
                        default=5e5, type=int)
    parser.add_argument('--save', help='Save the pre-trained model', action='store_true')
    parser.add_argument('--expl-noise', help='Exploration noise', default=0.1, type=float)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--discount', help='Discount factor gamma', default=0.99, type=float)
    parser.add_argument('--tau', help='Target network update rate', default=5e-3, type=float)
    parser.add_argument('--policy-noise',
                        help='STD of Gaussian noise added to the actions for '
                             'the exploration purposes',
                        default=0.2, type=float)
    parser.add_argument('--noise-clip', help='Maximum value of the Gaussian noise '
                                             'added to the actions (policy)')
    parser.add_argument('--policy-freq', help='Policy update frequency', default=2, type=int)
    parser_args = parser.parse_args()

    main(**parser_args.__dict__)
