import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorNetwork(nn.Module):
    def __init__(self, state_shape, action_shape, max_action):
        super().__init__()

        self.fc1 = nn.Linear(state_shape, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_shape)

        self.max_action = max_action

    def forward(self, state):
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        actions = self.max_action * T.tanh(self.fc3(state))

        return actions


class CriticNetwork(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
    
        # Defining the first Critic NN
        self.fc11 = nn.Linear(state_shape + action_shape, 400)
        self.fc12 = nn.Linear(400, 300)
        self.fc13 = nn.Linear(300, 1)
        # Defining the second Critic NN
        self.fc21 = nn.Linear(state_shape + action_shape, 400)
        self.fc22 = nn.Linear(400, 300)
        self.fc23 = nn.Linear(300, 1)

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
