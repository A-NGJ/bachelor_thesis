import numpy as np
import torch as T
from torch import nn

from network import (
    ActorNetwork,
    CriticNetwork
)
from replay_memory import ReplayBuffer
import util


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
        self.replay_memory = ReplayBuffer(state_shape, action_shape)
        self.loss = nn.MSELoss()

    def select_action(self, state):
        state = T.Tensor(state.reshape(1, -1)).to(self.device)
        action = self.actor.forward(state).cpu()
        return action.data.numpy().flatten()

    def train(self, iterations, batch_size=100,
              discount=0.99, tau=5e-3, policy_noise=0.2, noise_clip=0.5,
              policy_freq=2):
        for i in range(iterations):
            # Step4: We sample a batch of transitions (s, a, r, s') from the mem
            states, actions, rewards, states_, dones = \
                                      self.replay_memory.sample_buffer(batch_size)
            state = T.Tensor(states).to(self.device)
            action = T.Tensor(actions).to(self.device)
            reward = T.Tensor(rewards).to(self.device)
            state_ = T.Tensor(states_).to(self.device)
            done = T.tensor(dones, dtype=np.bool).to(self.device)
            # Step5: From the next state s', the Actor target plays the next action a'
            action_ = self.actor_target.forward(state_)
            # Step6: We add Gausian noise to this next action a' and we clamp it
            # in a range of values supported by the env
            noise = T.Tensor(actions).data.normal_(0, policy_noise).to(self.device)
            noise = noise.clamp(-noise_clip, noise_clip)
            action_ = (action_ + noise).clamp(-self.max_action, self.max_action)
            # Step7
            target_q1, target_q2 = self.critic_target.forward(state_, action_)
            # Step8
            target_q = T.min(target_q1, target_q2)
            # Step9
            target_q[done] = 0.0
            target_q = reward + (discount * target_q).detach()
            # Step10
            q1, q2 = self.critic.forward(state, action)
            # Step11
            critic_loss = (self.loss(q1, target_q) + self.loss(q2, target_q)).to(self.device)
            # Step12
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            # Step13
            if i % policy_freq == 0:
                actor_loss = -self.critic.q(1, state, self.actor.forward(state)).mean()
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                #step 14
                util.polyak_avg(self.actor.parameters(), self.actor_target.parameters(), tau)
                #step 15
                util.polyak_avg(self.critic.parameters(), self.critic_target.parameters(), tau)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
