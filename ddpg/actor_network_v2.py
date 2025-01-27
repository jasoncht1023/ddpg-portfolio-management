import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

# Actor / Policy Network / mu
# decide what to do based on the current state, outputs action values
class ActorNetwork(nn.Module):
    def __init__(self, learning_rate, n_actions, fc1_dims, fc2_dims, fc3_dims, name, chkpt_dir="tmp/ddpg"):
        super(ActorNetwork, self).__init__()
        self.n_actions = n_actions
        self.input_size = (n_actions-1) * 4 + n_actions + 1
        self.relu = nn.ReLU()
        self.checkpoint_file = os.path.join(chkpt_dir, name + "_ddpg")

        self.fc1 = nn.Linear(self.input_size, fc1_dims)
        self.bn1 = nn.LayerNorm(fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        # T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        # T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.bn2 = nn.LayerNorm(fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        # T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        # T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.bn3 = nn.LayerNorm(fc3_dims)
        f3 = 0.003
        # T.nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        # T.nn.init.uniform_(self.fc3.bias.data, -f3, f3)

        self.mu = nn.Linear(fc3_dims, self.n_actions)

        self.sigmoid = nn.Sigmoid()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
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
        # print("actor mu:", x)
        x = self.sigmoid(x)
        # print("actor sigmoid:", x)
        return x

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file): 
            print("... loading checkpoint ...")
            self.load_state_dict(T.load(self.checkpoint_file))

# for testing only
if __name__ == "__main__":
    n_actions = 10
    policy_net = ActorNetwork(
        learning_rate=1e-2, n_actions=n_actions, name="model name"
    )
    criterion = nn.CrossEntropyLoss()
    target = T.Tensor(
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
    ).to(policy_net.device)
    state_example = T.randn(4, 10, 10, 10).to(policy_net.device)
    for i in range(40):
        policy_net.optimizer.zero_grad()
        action = policy_net(state_example)
        loss = criterion(action, target)
        loss.backward()
        policy_net.optimizer.step()
    print(action)
