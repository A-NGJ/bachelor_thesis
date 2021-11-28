import os
import sys

import numpy as np
import torch as T
from torch import nn
import torch.nn.functional as F
from torch import optim


class DeepQNetworkBase(nn.Module):
    def __init__(self, name, input_dims, chkpt_dir):
        super().__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.conv1 = nn.Conv2d(input_dims[0], 32, 3, stride=1, padding=1)
        self.max_pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.max_pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.max_pool3 = nn.MaxPool2d(2, stride=2)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.max_pool1(dims)
        dims = self.conv2(dims)
        dims = self.max_pool2(dims)
        dims = self.conv3(dims)
        dims = self.max_pool3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        raise NotImplementedError()

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class DeepQNetwork(DeepQNetworkBase):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super().__init__(name, input_dims, chkpt_dir)
        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 256)
        self.fc2 = nn.Linear(256, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv1 = self.max_pool1(conv1)
        conv2 = F.relu(self.conv2(conv1))
        conv2 = self.max_pool2(conv2)
        conv3 = F.relu(self.conv3(conv2))
        conv3 = self.max_pool3(conv3)
        # conv3 shape is BS x n_filters x H x W
        conv_state = conv3.view(conv3.size()[0], -1)
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)

        return actions


class DuelingDeepQNetwork(DeepQNetworkBase):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super().__init__(name, input_dims, chkpt_dir)
        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 256)
        self.V = nn.Linear(256, 1)
        self.A = nn.Linear(256, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv1 = self.max_pool1(conv1)
        conv2 = F.relu(self.conv2(conv1))
        conv2 = self.max_pool2(conv2)
        conv3 = F.relu(self.conv3(conv2))
        conv3 = self.max_pool3(conv3)
        # conv3 shape is BS x n_filters x H x W
        conv_state = conv3.view(conv3.size()[0], -1)
        flat1 = F.relu(self.fc1(conv_state))
        V = self.V(flat1)
        A = self.A(flat1)

        return V, A
