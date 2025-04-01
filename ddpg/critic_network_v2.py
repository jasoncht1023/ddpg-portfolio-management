import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from .actor_network_v2 import ActorNetwork

# Critic / Q-value Network / Q
# evaluate state/action pairs
class CriticNetwork(nn.Module):
    def __init__(self, learning_rate, n_actions, lstm_size, name):
        super(CriticNetwork, self).__init__()
        self.name = name
        layer_dims = (n_actions-1) * 4 + n_actions * 2 + 1 
        self.relu = nn.ReLU()       

        self.lstm1 = nn.LSTM(layer_dims, lstm_size)
        self.__init_lstm(self.lstm1)

        self.lstm2 = nn.LSTM(lstm_size, lstm_size)
        self.__init_lstm(self.lstm2)

        self.fc = nn.Linear(lstm_size, layer_dims)
        self.bn = nn.LayerNorm(layer_dims)
        f1 = 0.004           
        nn.init.uniform_(self.fc.weight.data, -f1, f1)
        nn.init.uniform_(self.fc.bias.data, -f1, f1)

        self.q = nn.Linear(layer_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.device = T.device("cpu")
        self.to(self.device)

    def forward(self, state, action):
        x = T.cat((state, action), dim=-1)
        x = x.unsqueeze(0)
        output, (final_hidden_state, final_cell_state) = self.lstm1(x)
        output, (final_hidden_state, final_cell_state) = self.lstm2(final_hidden_state)
        x = final_hidden_state.squeeze(0)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        state_action_value = self.q(x)

        return state_action_value
    
    def __init_lstm(self, lstm_layer):
        for name, param in lstm_layer.named_parameters():
            if ("weight_ih" in name):                                     # Input-to-hidden weights
                nn.init.xavier_uniform_(param)
            elif ("weight_hh" in name):                                   # Hidden-to-hidden (recurrent) weights
                nn.init.orthogonal_(param)
            elif ("bias" in name): 
                param.data.fill_(0)
                hidden_size = param.shape[0] // 4 
                param.data[hidden_size:hidden_size * 2].fill_(1)  

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