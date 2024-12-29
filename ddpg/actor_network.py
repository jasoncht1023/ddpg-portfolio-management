import torch as T
import torch.nn as nn
import torch.optim as optim
import tensorly as tl
from tensorly.decomposition import tucker
from functools import reduce
import operator


# Actor / Policy Network / mu
# decide what to do based on the current state, outputs action values
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, name, chkpt_dir="tmp/ddpg"):
        super(ActorNetwork, self).__init__()
        self.tucker_dimension = [8, 6, 6, 6]
        self.n_actions = n_actions
        self.relu = nn.ReLU()
        self.conv3d = nn.Conv3d(in_channels=4, out_channels=32, kernel_size=(1, 3, 1))
        self.fc = nn.Linear(
            reduce(operator.mul, self.tucker_dimension, 1), self.n_actions
        )
        self.softmax = nn.Softmax(dim=-1)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)
        tl.set_backend("pytorch")

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cuda:1")
        self.to(self.device)

    def forward(self, x):
        x = self.conv3d(x)
        x = self.relu(x)
        core, factors = tucker(x, rank=self.tucker_dimension)  # can be change
        core.requires_grad_(True)
        x = T.flatten(core)
        x = self.fc(x)
        x = self.softmax(x)
        x = x / x.sum()
        return x

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))


# for testing only
if __name__ == "__main__":
    n_actions = 10
    policy_net = ActorNetwork(n_actions=n_actions, name="model name")
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
    input = T.randn(4, 10, 10, 10).to(policy_net.device)
    for i in range(40):
        policy_net.optimizer.zero_grad()
        output = policy_net(input)
        loss = criterion(output, target)
        loss.backward()
        policy_net.optimizer.step()
    print(output)
