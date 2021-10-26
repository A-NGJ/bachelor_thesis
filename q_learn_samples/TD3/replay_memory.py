import numpy as np


class ReplayBuffer:
    def __init__(self, state_shape, action_shape, max_size=1e6):
        self.mem_size = int(max_size)

        self.state_memory = np.zeros((self.mem_size, *state_shape),
                                      dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *state_shape),
                                          dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, *action_shape),
                                      dtype=np.float64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.mem_cntr = 0

    def store_transiton(self, transition):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = transition[0]
        self.new_state_memory[index] = transition[1]
        self.action_memory[index] = transition[2]
        self.reward_memory[index] = transition[3]
        self.terminal_memory[index] = transition[4]
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards.reshape(-1, 1), states_, dones.reshape(-1, 1)

class ReplayBufferLegacy(object):

    def __init__(self, x, y, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def store_transiton(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample_buffer(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind: 
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        return np.array(batch_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_next_states), np.array(batch_dones).reshape(-1, 1)

