import collections
import functools
import os
import yaml

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import rospy

from enums import Dir
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment


def make_dir(path):
    def decorator_make_dir(func):
        @functools.wraps(func)
        def wrapper_make_dir(*args, **kwargs):
            if not os.path.exists(path):
                os.makedirs(path)
            func(*args, **kwargs)
        return wrapper_make_dir
    return decorator_make_dir


@make_dir(Dir.PLOT.value)
def plot_learning_curve(x, scores, epsilons, filename, avg_range, *, extension='.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111, label='1')
    ax2 = fig.add_subplot(111, label='2', frame_on=False)

    ax.plot(x, epsilons, color='C0')
    ax.set_xlabel('Training Steps', color='C0')
    ax.set_ylabel('Epsilon', color='C0')
    ax.tick_params(axis='x', colors='C0')
    ax.tick_params(axis='y', colors='C0')

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-avg_range):(t+1)])

    ax2.scatter(x, running_avg, color='C1')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color='C1')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors='C1')

    dir_ = f'{Dir.PLOT.value}{filename}{extension}'
    plt.savefig(dir_)


@make_dir(Dir.PLOT.value)
def plot_history(param_name, values, filename, *, extenstion='.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(values)), values)

    ax.set_xlabel('Epochs', color='C0')
    ax.set_ylabel(param_name, color='C0')
    ax.tick_params(axis='x', colors='C0')
    ax.tick_params(axis='y', colors='C0')

    dir_ = f'{Dir.PLOT.value}{filename}_{param_name}{extenstion}'
    plt.savefig(dir_)


@make_dir(Dir.PARAMS.value)
def save_parameters(parameters, filename, *, extension='.yaml'):
    dir_ = f'{Dir.PARAMS.value}{filename}{extension}'
    with open(dir_, 'w', encoding='utf-8') as yml:
        yaml.dump(parameters, yml)


@make_dir(Dir.RESULTS.value)
def save_results(results, filename, *, extension='.npy'):
    dir_ = f'{Dir.RESULTS.value}{filename}{extension}'
    np.save(dir_, results)


@make_dir(Dir.TRACK.value)
def plot_trace(algo):
    reds = np.load(os.path.join(Dir.TRACK.value, 'reds.npy'))
    greens = np.load(os.path.join(Dir.TRACK.value, 'greens.npy'))
    dir_ = os.path.join(Dir.TRACK.value, algo)

    fig = plt.figure()
    fig.set_size_inches(10, 7)
    ax = fig.add_subplot(111)
    ax.scatter(reds[:, 0], reds[:, 1], color='r')
    ax.scatter(greens[:, 0], greens[:, 1], color='g')
    ax.vlines(x=33, color='#000000', label='Finish', ymin=53, ymax=71)
    ax.vlines(x=100, color='#808080', label='Start', ymin=56, ymax=74)

    for i in range(5):
        trace = np.load(os.path.join(dir_, f'route_{i}.npy'))
        ax.plot(trace[1:, 0], trace[1:, 1], color=f'C{i}', linewidth=2.0)

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.legend()
    plt.savefig(os.path.join(dir_, 'trace.png'))


class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env=None, repeat=4, clip_reward=False, no_ops=0,
                 fire_first=False):
        super().__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first

    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            if self.clip_reward:
                reward = np.clip(np.array([reward]), -1, 1)[0]
            t_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, done, info

    def reset(self):
        obs = self.env.reset()
        no_ops = np.random.randint(1, self.no_ops+1) if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _, _ = self.env.step(1)

        self.frame_buffer = np.zeros_like((2, self.shape))
        self.frame_buffer[0] = obs

        return obs


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super().__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=self.shape, dtype=np.float32)

    # def observation(self, observation):
    #     images = observation[-2:]
    #     for i in range(2):
    #         images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
    #         images[i] = images[i][:400, ...]
    #         images[i] = cv2.resize(images[i], self.shape[:-1], interpolation=cv2.INTER_AREA)

    #     new_image = np.concatenate(
    #         [img[:, :img.shape[1]//2] for img in images],
    #         axis=1
    #     )

        # new_frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        # resized_screen = cv2.resize(new_frame, self.shape[1:],
        #                            interpolation=cv2.INTER_AREA)

        # new_image = np.array(new_image, dtype=np.uint8).reshape(self.shape)
        # new_image = new_image / 255.0

        # return new_image

    def observation(self, observation):
        return observation


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super().__init__(env)
        # self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(repeat, axis=0),
            env.observation_space.high.repeat(repeat, axis=0),
            dtype=np.float32)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self, **kwargs):
        self.stack.clear()
        obs = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(obs)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)


def make_env(env_name, shape=(64, 80, 3), repeat=5, clip_rewards=False,
             no_ops=0, fire_first=False):
    #env = gym.make(env_name)
    env = StartOpenAI_ROS_Environment(env_name)
    #env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)

    return env


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
