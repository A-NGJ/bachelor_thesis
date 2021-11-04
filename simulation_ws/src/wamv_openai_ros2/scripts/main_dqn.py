#!/usr/bin/env python
import cv2
from datetime import datetime
import numpy as np
import os
import torch as T

import rospy

from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
from agents.dueling_dqn_agent import (
    DuelingDDQNAgent,
    DuelingDQNAgent
)
from enums import Dir
from deep_q_network import (
    DeepQNetwork,
    DuelingDeepQNetwork
)
from util import (
    make_env,
    plot_learning_curve,
    plot_history,
    save_parameters,
    save_results,
    Namespace
)

#pylint: disable=too-many-locals
def main(agent_class, network_class, algo, args):
    T.cuda.empty_cache()
    rospy.init_node('wamv_dqn', anonymous=True, log_level=rospy.INFO)

    env = make_env(args.env_name)
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    best_score = -np.inf
    agent = agent_class(network=network_class, gamma=args.gamma,
                        epsilon=args.epsilon, lr=args.lr,
                        input_dims=env.observation_space.shape,
                        n_actions=env.action_space.n, mem_size=args.mem_size, eps_min=args.eps_min,
                        batch_size=args.batch_size, replace=args.replace, eps_dec=args.eps_dec,
                        chkpt_dir=Dir.CHECKPOINT.value, algo=algo,
                        env_name=f'{args.env_name}_{args.env_type}')

    if args.load_checkpoint:
        agent.load_models()

    # saving to video
    # env = wrappers.Monitor(env, 'tmp/video',
    #                        video_callable=lambda episode_id: True,
    #                        force=True) <- overriding previous episode

    date_ = datetime.now().strftime('%m%d_%H%M')

    fname = f'{agent.algo}_{agent.env_name}_{date_}'
    if args.load_checkpoint:
        fname = fname + '_eval'

    n_steps = 0
    scores, eps_history, steps_array, loss_history, q_history = [], [], [], [], []

    for i in range(args.n_games):
        done = False
        score = 0
        observation = env.reset()
        agent.zero_loss_history()

        game_steps = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            score += reward

            if not args.load_checkpoint:
                agent.store_transition(observation, action, reward,
                                       observation_, int(done))
                agent.learn()
            observation = observation_
            n_steps += 1
            game_steps += 1
            if game_steps > args.max_steps:
                break

        if not args.load_checkpoint:
            loss_history.append(np.mean(agent.loss_history))
            q_history.append(np.mean(agent.q_history))
        scores.append(score)
        steps_array.append(n_steps)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-args.n_games//5:])

        print('='*80)
        print(f'episode {i} score {score:.2f} '
              f'average_score {avg_score:.1f} '
              f'best_score {best_score:.1f} '
              f'epsilon {agent.epsilon:.2f} '
              f'steps {n_steps}')
        print('='*80)

        if avg_score > best_score:
            if not args.load_checkpoint:
                agent.save_models()
            best_score = avg_score

    if not args.load_checkpoint:
        plot_history('loss', loss_history, fname)
        plot_history('qval', q_history, fname)
        plot_learning_curve(steps_array, scores, eps_history,
                        fname, avg_range=args.n_games//5)
        save_parameters(args.__dict__, fname)
        for name, result in zip(('scores', 'eps', 'loss'), (scores, eps_history, loss_history)):
            save_results(result, f'{fname}_{name}')


if __name__=='__main__':
    algorithm = rospy.get_param('/wamv/algorithm')
    gamma = float(rospy.get_param('/wamv/gamma'))
    epsilon = float(rospy.get_param('/wamv/epsilon'))
    lr = float(rospy.get_param('/wamv/lr'))
    mem_size = int(rospy.get_param('/wamv/mem_size'))
    eps_min = float(rospy.get_param('/wamv/eps_min'))
    batch_size = int(rospy.get_param('/wamv/batch_size'))
    replace = int(rospy.get_param('/wamv/replace'))
    eps_dec = float(rospy.get_param('/wamv/eps_dec'))
    load_checkpoint = bool(rospy.get_param('/wamv/load_checkpoint'))
    env_name = rospy.get_param(
        '/wamv/task_and_robot_environment_name'
    )
    env_type = rospy.get_param('/wamv/environment_type')
    n_games = int(rospy.get_param('/wamv/n_games'))
    max_steps = int(rospy.get_param('/wamv/max_steps'))

    if load_checkpoint:
        epsilon = 0.0
        eps_dec = 0.0

    args = Namespace(
        gamma=gamma,
        epsilon=epsilon,
        lr=lr,
        mem_size=mem_size,
        eps_min=eps_min,
        batch_size=batch_size,
        replace=replace,
        eps_dec=eps_dec,
        load_checkpoint=load_checkpoint,
        env_name=env_name,
        env_type=env_type,
        n_games=n_games,
        max_steps=max_steps
    )

    if algorithm == 'dqn':
        algo = 'DQNAgent'
        agent_class = DQNAgent
        network_class = DeepQNetwork
    elif algorithm == 'ddqn':
        algo = 'DDQNAgent'
        agent_class = DDQNAgent
        network_class = DeepQNetwork
    elif algorithm == 'dueling-dqn':
        algo = 'DuelingDQNAgent'
        agent_class = DuelingDQNAgent
        network_class = DuelingDeepQNetwork
    elif algorithm == 'dueling-ddqn':
        algo = 'DuelingDDQNAgent'
        agent_class = DuelingDDQNAgent
        network_class = DuelingDeepQNetwork

    # for multiple gpus
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(agent_class, network_class, algo, args)
