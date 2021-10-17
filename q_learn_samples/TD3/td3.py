import torch as T

from network import (
    ActorNetwork,
    CriticNetwork
)


class TD3:
    def __init__(self, state_shape, action_shape, max_action):
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.actor = ActorNetwork(state_shape, action_shape,
                                  max_action).to(self.device)
        self.actor_target = ActorNetwork(state_shape, action_shape,
                                         max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = CriticNetwork(state_shape, action_shape).to(self.device)
        self.critic_target = CriticNetwork(state_shape, action_shape).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.max_action = max_action

    def select_action(self, state):
        state = T.Tensor(state.reshape(1, -1)).to(self.device)
        action = self.actor.forward(state).cpu()
        return action.data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100,
              discount=0.99, tau=5e-3, policy_noise=0.2):