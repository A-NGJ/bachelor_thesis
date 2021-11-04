import numpy as np
import torch as T

from agents.base_agent import BaseAgent


class DuelingDQNAgent(BaseAgent):
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True))).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()


class DuelingDDQNAgent(DuelingDQNAgent):
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next[indices, max_actions]
        self.q_history.append(q_target.cpu().detach().numpy())

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.loss_history.append(loss.cpu().detach().numpy())
        for param in self.q_eval.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
