import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from .actor_network import ActorNetwork

# Critic / Q-value Network / Q
# evaluate state/action pairs
class CriticNetwork(nn.Module):
    def __init__(self, learning_rate, n_actions, lstm_size, name, chkpt_dir="tmp/ddpg"):
        super(CriticNetwork, self).__init__()
        layer_dims = (n_actions-1) * 4 + n_actions * 2 + 1 
        self.relu = nn.ReLU()
        self.checkpoint_file = os.path.join(chkpt_dir, name + "_ddpg")

        self.lstm1 = nn.LSTM(layer_dims, lstm_size, dropout=0.35)
        self.lstm2 = nn.LSTM(lstm_size, lstm_size, dropout=0.35)
        self.fc = nn.Linear(lstm_size, layer_dims)
        self.bn = nn.LayerNorm(layer_dims)
        f1 = 1./np.sqrt(self.fc.weight.data.size()[0])
        # T.nn.init.uniform_(self.fc.weight.data, -f1, f1)
        # T.nn.init.uniform_(self.fc.bias.data, -f1, f1)

        self.q = nn.Linear(layer_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        x = T.cat((state, action), dim=-1)
        x = x.unsqueeze(0)
        # print("input shape:", x.shape)
        output, (h_n, c_n) = self.lstm1(x)
        # print("lstm1 output shape:", h_n.shape)
        output, (h_n, c_n) = self.lstm2(h_n)
        # print("lstm2 output shape:", h_n.shape)
        x = h_n.squeeze(0)
        # print("lstm2 output shape after squeeze:", x.shape)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        state_action_value = self.q(x)

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
