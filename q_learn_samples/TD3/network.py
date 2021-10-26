from abc import ABC
import os

import torch as T
from torch import nn
import torch.nn.functional as F
from torch import optim

class BaseNetwork(nn.Module, ABC):
    def __init__(self, name='', chckpt_dir='models/'):
        super().__init__()
        self.chckpt_dir = chckpt_dir
        self.name = f'{name}_{self.__name__}.pth'
        self.checkpoint_file = os.path.join(self.chckpt_dir, self.name)

    def save_checkpoint(self):
        if not os.path.exists(self.chckpt_dir):
            os.mkdir(self.chckpt_dir)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(BaseNetwork):
    __name__ = 'ActorNetwork'

    def __init__(self, state_shape, action_shape, max_action, name='', chckpt_dir='models/'):
        super().__init__(name, chckpt_dir)

        self.fc1 = nn.Linear(state_shape, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_shape)

        self.optimizer = optim.Adam(self.parameters())

        self.max_action = max_action

    def forward(self, state):
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        actions = self.max_action * T.tanh(self.fc3(state))

        return actions


class CriticNetwork(BaseNetwork):
    __name__ = 'CriticNetwork'

    def __init__(self, state_shape, action_shape, name='', chckpt_dir='models/'):
        super().__init__(name, chckpt_dir)

        # Defining the first Critic NN
        self.fc11 = nn.Linear(state_shape + action_shape, 400)
        self.fc12 = nn.Linear(400, 300)
        self.fc13 = nn.Linear(300, 1)
        # Defining the second Critic NN
        self.fc21 = nn.Linear(state_shape + action_shape, 400)
        self.fc22 = nn.Linear(400, 300)
        self.fc23 = nn.Linear(300, 1)

        self.optimizer = optim.Adam(self.parameters())

    def _iter_q(self, n):
        for i in range(1, 4):
            yield getattr(self, f'fc{n}{i}')

    def q(self, n, state, action):
        state_action = T.cat([state, action], dim=1)
        dqn = self._iter_q(n)
        state_action = F.relu(next(dqn)(state_action))
        state_action = F.relu(next(dqn)(state_action))
        return next(dqn)(state_action)

    def forward(self, state, action):
        return self.q(1, state, action), self.q(2, state, action)
