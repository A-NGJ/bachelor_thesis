import argparse
import os
import time

import numpy as np
import gym
from gym import wrappers
import pybullet_envs

from td3 import TD3
import util


def main(**kwargs):
    filename = f'TD3_{kwargs["env"]}_{int(kwargs["seed"])}'
    models_dir = './pytorch_models'
    results_dir = './results'

    print('-'*20)
    print(f'Settings: {filename}')
    print('-'*20)

    if kwargs["save"] and not os.path.exists('./pytorch_models'):
        os.mkdir(models_dir)

    env = gym.make(kwargs["env"])
    util.set_seed(kwargs['seed'], env)

    state_shape, action_shape, max_action = util.get_shapes(env)

    policy = TD3((state_shape), (action_shape), max_action,
                 name=filename, chckpt_dir=models_dir)

    workdir = util.mkdir('exp', 'brs')
    monitor_dir = util.mkdir(workdir, 'monitor')

    if kwargs['save_video']:
        env = wrappers.Monitor(env, monitor_dir, force=True)
        env.reset()

    if kwargs['load']:
        policy.load_models()
        util.eval_policy(env, policy)
        return

    evaluations = [util.eval_policy(env, policy)]

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_n = 0
    done = True

    while total_timesteps < kwargs['max_timesteps']:

        if done:

            if total_timesteps != 0:
                print(f'Total Timesteps: {total_timesteps} '
                      f'Episode Num: {episode_n} '
                      f'Reward {episode_reward}')
                policy.train(episode_timesteps, **kwargs)

            if timesteps_since_eval >= kwargs['eval_freq']:
                timesteps_since_eval %= kwargs['eval_freq']
                evaluations.append(util.eval_policy(env, policy))
                if kwargs['save']:
                    policy.save_models()
                    np.save(os.path.join(results_dir, filename), evaluations)

            obs = env.reset()

            done = False

            episode_reward = 0
            episode_timesteps = 0
            episode_n += 1

        if total_timesteps < kwargs['start_timesteps']:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(obs))
            if kwargs['expl_noise'] != 0:
                action = (action + np.random.normal(0, kwargs['expl_noise'],
                                                    size=env.action_space.shape[0])
                         ).clip(env.action_space.low, env.action_space.high)

        new_obs, reward, done, _ = env.step(action)

        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)

        episode_reward += reward

        policy.replay_memory.store_transiton((obs, new_obs, action, reward, done_bool))

        obs = new_obs
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    evaluations.append(util.eval_policy(env, policy))
    if kwargs['save']:
        policy.save_models()
    np.save(os.path.join(results_dir, filename), evaluations)

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
                                             'added to the actions (policy)',
                        default=0.5)
    parser.add_argument('--policy-freq', help='Policy update frequency', default=2, type=int)
    parser.add_argument('--load', help='Load pre-trained model', action='store_true')
    parser.add_argument('--save-video', action='store_true')
    parser_args = parser.parse_args()

    main(**parser_args.__dict__)
