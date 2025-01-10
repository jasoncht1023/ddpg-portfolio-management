import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tensorly as tl
from tensorly.decomposition import tucker
from functools import reduce
import operator
import os
from .actor_network import ActorNetwork


# Critic / Q-value Network / Q
# evaluate state/action pairs
class CriticNetwork(nn.Module):
    def __init__(self, learning_rate, n_actions, name, chkpt_dir="tmp/ddpg"):
        super(CriticNetwork, self).__init__()
        self.tucker_dimension = [8, 2, 6, 2]
        self.n_actions = n_actions
        self.relu = nn.ReLU()
        self.checkpoint_file = os.path.join(chkpt_dir, name + "_ddpg")

        # for state_value
        self.conv3d = nn.Conv3d(in_channels=4, out_channels=32, kernel_size=(1, 3, 1))
        self.fc = nn.Linear(reduce(operator.mul, self.tucker_dimension, 1), self.n_actions)
        self.softmax = nn.Softmax(dim=-1)

        # for action_value
        self.action_value = nn.Linear(n_actions, reduce(operator.mul, self.tucker_dimension, 1))
        self.q = nn.Linear(reduce(operator.mul, self.tucker_dimension, 1), 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        tl.set_backend("pytorch")

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cuda:1")
        self.to(self.device)

    def forward(self, state, action):
        state_value = self.conv3d(state)
        state_value = self.relu(state_value)
        core, factors = tucker(state_value, rank=self.tucker_dimension)  # can be change
        core.requires_grad_(True)
        state_value = T.flatten(core)

        action_value = self.action_value(action)
        action_value = self.relu(action_value)

        state_action_value = T.add(state_value, action_value)
        state_action_value = F.relu(state_action_value)  # might need to change, relu then add vs add then relu
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file): 
            print("... loading checkpoint ...")
            self.load_state_dict(T.load(self.checkpoint_file))    

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
