import torch
import torch.nn as nn
import torch.optim as optim
import tensorly as tl
from tensorly.decomposition import tucker
from collections import defaultdict
import pandas as pd
import yfinance as yf
import numpy as np
from functools import reduce
import operator


class ActorNetwork(nn.Module):
    def __init__(self, name, chkpt_dir="tmp/ddpg"):
        super(ActorNetwork, self).__init__()
        self.tucker_dimension = [8, 6, 6, 6]
        self.num_of_actions = 10
        self.relu = nn.ReLU()
        self.conv3d = nn.Conv3d(in_channels=4, out_channels=32, kernel_size=(1, 3, 1))
        self.fc = nn.Linear(
            reduce(operator.mul, self.tucker_dimension, 1), self.num_of_actions
        )
        self.softmax = nn.Softmax(dim=-1)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)
        tl.set_backend("pytorch")

    def forward(self, x):
        x = self.conv3d(x)
        x = self.relu(x)
        core, factors = tucker(x, rank=self.tucker_dimension)  # can be change
        core.requires_grad_(True)
        x = torch.flatten(core)
        x = self.fc(x)
        x = self.softmax(x)
        x = x / x.sum()
        return x

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(torch.load(self.checkpoint_file))


# for testing only
if __name__ == "__main__":

    def get_params(model):
        for name, param in model.named_parameters():
            if param.requires_grad and name == "conv3d.bias":
                return param.data

    criterion = nn.CrossEntropyLoss()
    policy_net = ActorNetwork("model name")
    for name, param in policy_net.named_parameters():
        print(name)

    target = torch.Tensor(
        [
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
        ]
    )
    input = torch.randn(4, 10, 10, 10)
    print(get_params(policy_net))
    for i in range(40):
        policy_net.optimizer.zero_grad()
        output = policy_net(input)
        loss = criterion(output, target)
        loss.backward()
        policy_net.optimizer.step()
    print(get_params(policy_net))
    print(output)
