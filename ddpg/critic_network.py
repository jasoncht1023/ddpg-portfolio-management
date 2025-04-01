import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tensorly as tl
import numpy as np
import os
from .actor_network import ActorNetwork

# Critic / Q-value Network / Q
# evaluate state/action pairs
class CriticNetwork(nn.Module):
    def __init__(self, learning_rate, n_actions, input_dims, fc_dims, name):
        super(CriticNetwork, self).__init__()
        # self.tucker_dimension = [8, 2, 6, 2]
        self.name = name
        self.n_actions = n_actions
        self.input_size = 32 * input_dims[1] * (input_dims[2]-2) * input_dims[3]
        self.fc1_dims = fc_dims
        self.relu = nn.ReLU()

        # for state_value
        self.conv3d = nn.Conv3d(in_channels=4, out_channels=32, kernel_size=(1, 3, 1))
        # self.fc = nn.Linear(reduce(operator.mul, self.tucker_dimension, 1), 300)
        self.fc1 = nn.Linear(self.input_size, self.fc1_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[1])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)


        self.fc2 = nn.Linear(self.fc1_dims, 300)
        self.bn2 = nn.LayerNorm(300)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[1])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        self.action_value = nn.Linear(self.n_actions, 300)
        self.q = nn.Linear(300, 1)
        f3 = 0.003
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.softmax = nn.Softmax(dim=-1)

        # for action_value
        # self.action_value = nn.Linear(n_actions, reduce(operator.mul, self.tucker_dimension, 1))
        # self.q = nn.Linear(reduce(operator.mul, self.tucker_dimension, 1), 1)
        # self.action_value = nn.Linear(n_actions, self.fc_dims)
        # self.q = nn.Linear(self.fc_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        tl.set_backend("pytorch")

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        state_value = self.conv3d(state)
        state_value = self.relu(state_value)
        # core, factors = tucker(state_value, rank=self.tucker_dimension)  # can be change
        # core.requires_grad_(True)
        # state_value = T.flatten(core)
        state_value = T.flatten(state_value)

        state_value = self.fc1(state_value)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = self.action_value(action)
        action_value = self.relu(action_value)

        state_action_value = T.add(state_value, action_value)
        state_action_value = F.relu(state_action_value)  # might need to change, relu then add vs add then relu
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self, saving_dir):
        print("... saving checkpoint ...")
        checkpoint_file = os.path.join(saving_dir, self.name + "_ddpg")
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, loading_dir):
        checkpoint_file = os.path.join(loading_dir, self.name + "_ddpg")
        if os.path.exists(checkpoint_file): 
            print("... loading checkpoint ...")
            self.load_state_dict(T.load(checkpoint_file))

# for testing only
if __name__ == "__main__":
    learning_rate = 1e-2
    actor_net = ActorNetwork(
        learning_rate=learning_rate, n_actions=10, name="actor_model_test"
    )
    critic_net = CriticNetwork(
        learning_rate=learning_rate, n_actions=10, name="critic_model_test"
    )

    state_example = T.randn(4, 10, 10, 10).to(actor_net.device)
    action = actor_net(state_example)
    action_state_value = critic_net(state_example, action)
    print(action_state_value)
