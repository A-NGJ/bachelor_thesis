import numpy as np


class ReplayBuffer:
    def __init__(self, state_shape, action_shape, max_size=1e6):
        self.mem_size = max_size
        self.state_memory = np.zeros((self.mem_size, *state_shape),
                                      dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *state_shape),
                                          dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, *action_shape,
                                      dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.mem_cntr = 0

    def _iter_mem(self):
        yield self.state_memory
        yield self.new_state_memory
        yield self.action_memory
        yield self.reward_memory
        yield self.terminal_memory

    def store_transiton(self, transition):
        index = self.mem_cntr % self.mem_size
        for i, mem in enumerate(self._iter_mem()):
            mem[index] = transition[i]
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones
