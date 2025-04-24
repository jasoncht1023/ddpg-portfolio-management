import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

# Actor / Policy Network / mu
# decide what to do based on the current state, outputs action values
class ActorNetworkFC(nn.Module):
    def __init__(self, learning_rate, n_actions, fc1_dims, fc2_dims, fc3_dims, name):
        super(ActorNetworkFC, self).__init__()
        self.name = name
        self.n_actions = n_actions
        self.input_size = (n_actions - 1) * 7 + n_actions
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(self.input_size, fc1_dims)
        self.bn1 = nn.LayerNorm(fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[1])                 # Square root of the fan-in
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.bn2 = nn.LayerNorm(fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[1])                 # Square root of the fan-in
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.bn3 = nn.LayerNorm(fc3_dims)
        f3 = 0.003
        nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        nn.init.uniform_(self.fc3.bias.data, -f3, f3)

        self.mu = nn.Linear(fc3_dims, self.n_actions)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.device = T.device("cpu")
        self.to(self.device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.mu(x)
        x = self.softmax(x)
        return x

    def save_checkpoint(self, saving_dir):
        print("... saving checkpoint ...")
        checkpoint_file = os.path.join(saving_dir, self.name + "_ddpg")
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, loading_dir, is_model_zipped):
        if (is_model_zipped == True):
            checkpoint_file = os.path.join(loading_dir, self.name + "_ddpg.zip")
        else:
            checkpoint_file = os.path.join(loading_dir, self.name + "_ddpg")
        if os.path.exists(checkpoint_file): 
            print("... loading checkpoint ...")
            self.load_state_dict(T.load(checkpoint_file))
