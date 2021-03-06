import argparse
from datetime import datetime
import numpy as np

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
    save_parameters
)


def main(agent_class, network_class, algo, args):
    env = make_env('PongNoFrameskip-v4')
    best_score = -np.inf
    load_checkpoint = False
    n_games = 50
    agent = agent_class(network=network_class, gamma=0.99, epsilon=1.0, lr=1e-4,
                        input_dims=(env.observation_space.shape),
                        n_actions=env.action_space.n, mem_size=40000, eps_min=0.0,
                        batch_size=32, replace=1000, eps_dec=1e-5,
                        chkpt_dir='models/', algo=algo,
                        env_name='PongNoFrameskip-v4')

    parameters = {
        'env_name': 'PongNoFrameskip-v4',
        'n_games': n_games,
        'algorithm': args.algorithm,
        'gamma': 0.99,
        'epsilon': 1.0,
        'lr': 0.0001,
        'mem_size': 40000,
        'eps_min': 0.0,
        'batch_size': 32,
        'replace': 1000,
        'eps_dec': 0.00001
    }

    if load_checkpoint:
        agent.load_models()

    # saving to video
    # env = wrappers.Monitor(env, 'tmp/video',
    #                        video_callable=lambda episode_id: True,
    #                        force=True) <- overriding previous episode

    date_ = datetime.now().strftime('%m%d_%H%M')

    fname = f'{agent.algo}_{agent.env_name}_{date_}'

    n_steps = 0
    scores, eps_history, steps_array, loss_history = [], [], [], []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        agent.zero_loss_history()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if args.render:
                env.render()
            if not load_checkpoint:
                agent.store_transition(observation, action, reward,
                                       observation_, int(done))
                agent.learn()
            observation = observation_
            n_steps += 1

        loss_history.append(np.mean(agent.loss_history))
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print(f'episode {i} score {score} '
              f'average_score {avg_score:.1f} '
              f'best_score {best_score:.1f} '
              f'epsilon {agent.epsilon:.2f} '
              f'steps {n_steps}')

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    plot_loss_history(loss_history, fname)
    plot_learning_curve(steps_array, scores, eps_history, fname)
    save_parameters(parameters, fname)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', choices=['dueling-ddqn', 'dueling-dqn', 'dqn', 'ddqn'], type=str)
    parser.add_argument('-r', '--render', action='store_true')
    args = parser.parse_args()
    if args.algorithm == 'dqn':
        algo = 'DQNAgent'
        agent_class = DQNAgent
        network_class = DeepQNetwork
    elif args.algorithm == 'ddqn':
        algo = 'DDQNAgent'
        agent_class = DDQNAgent
        network_class = DeepQNetwork
    elif args.algorithm == 'dueling-dqn':
        algo = 'DuelingDQNAgent'
        agent_class = DuelingDQNAgent
        network_class = DuelingDeepQNetwork
    elif args.algorithm == 'dueling-ddqn':
        algo = 'DuelingDDQNAgent'
        agent_class = DuelingDDQNAgent
        network_class = DuelingDeepQNetwork

    # for multiple gpus
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(agent_class, network_class, algo, args)
