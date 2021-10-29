#!/usr/bin/env python
import numpy as np

import rospy

from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
from agents.dueling_dqn_agent import (
    DuelingDDQNAgent,
    DuelingDQNAgent
)
from deep_q_network import (
    DeepQNetwork,
    DuelingDeepQNetwork
)
from util import (
    make_env,
    plot_learning_curve,
    plot_loss_history,
    Namespace
)

def main(agent_class, network_class, algo, args):
    rospy.init_node('wamv_dqn', anonymous=True, log_level=rospy.INFO)

    env = make_env(args.env_name)
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    best_score = -np.inf
    agent = agent_class(network=network_class, gamma=args.gamma,
                        epsilon=args.epsilon, lr=args.lr,
                        input_dims=(1, 84, 84),
                        n_actions=env.action_space.n, mem_size=args.mem_size, eps_min=args.eps_min,
                        batch_size=args.batch_size, replace=args.replace, eps_dec=args.eps_dec,
                        chkpt_dir=args.chkpt_dir, algo=algo,
                        env_name=args.env_name)

    if args.load_checkpoint:
        agent.load_models()

    # saving to video
    # env = wrappers.Monitor(env, 'tmp/video',
    #                        video_callable=lambda episode_id: True,
    #                        force=True) <- overriding previous episode

    fname = f'{agent.algo}_{agent.env_name}_lr{agent.lr}__{n_games}games'
    figure_file = f'plots/{fname}.png'

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(args.n_games):
        done = False
        score = 0
        observation = env.reset()

        game_steps = 0
        while not done:
            action = agent.choose_action(observation[2])
            observation_, reward, done, info = env.step(action)
            score += reward

            if not args.load_checkpoint:
                agent.store_transition(observation[2], action, reward,
                                       observation_[2], int(done))
                agent.learn()
            observation = observation_
            n_steps += 1
            game_steps += 1
        scores.append(score)
        steps_array.append(n_steps)
        if game_steps > args.max_steps:
            break

        avg_score = np.mean(scores[-100:])
        print(f'episode {i} score {score} '
              f'average_score {avg_score:.1f} '
              f'best_score {best_score:.1f} '
              f'epsilon {agent.epsilon:.2f} '
              f'steps {n_steps}')

        if avg_score > best_score:
            if not args.load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)


    plot_learning_curve(steps_array, scores, eps_history, figure_file)
    plot_loss_history(agent.loss_history)


if __name__=='__main__':
    algorithm = rospy.get_param('/wamv/algorithm')
    gamma = np.float32(rospy.get_param('/wamv/gamma'))
    epsilon = np.float32(rospy.get_param('/wamv/epsilon'))
    lr = np.float32(rospy.get_param('/wamv/lr'))
    mem_size = np.int(rospy.get_param('/wamv/mem_size'))
    eps_min = np.float32(rospy.get_param('/wamv/eps_min'))
    batch_size = np.int(rospy.get_param('/wamv/batch_size'))
    replace = np.int(rospy.get_param('/wamv/replace'))
    eps_dec = np.float32(rospy.get_param('/wamv/eps_dec'))
    load_checkpoint = np.bool(rospy.get_param('/wamv/load_checkpoint'))
    env_name = rospy.get_param(
        '/wamv/task_and_robot_environment_name'
    )
    chkpt_dir = rospy.get_param('/results/chkpt_dir')
    n_games = np.int(rospy.get_param('/wamv/n_games'))
    max_steps = np.int(rospy.get_param('/wamv/max_steps'))
    # gamma = 0.99
    # epsilon = 0.1
    # eps_min = 0.0
    # n_games = 10
    # n_steps = 10000
    # algorithm = 'dqn'
    # lr = 0.0001
    # mem_size = 40000
    # batch_size = 32
    # replace = 1000
    # eps_dec = 0.00001
    # load_checkpoint = False
    # env_name = 'WamvNavTwoSetsBuoys-v0'
    # chkpt_dir = '/home/angj/priv/bachelor_thesis/simulation_ws/models/'

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
        chkpt_dir=chkpt_dir,
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
