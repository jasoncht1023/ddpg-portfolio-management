import numpy as np


# Replay buffer stores previous experiences, sample a mini-batch at each time step to update the weights of the neural networks
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state = np.zeros((self.mem_size, *input_shape))
        self.new_state = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state[index] = state
        self.new_state[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done  # For bellman equation, to multiply whether or not the episode is over
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        state = self.state[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_state = self.new_state[batch]
        terminal = self.terminal_memory[batch]

        return state, actions, rewards, new_state, terminal