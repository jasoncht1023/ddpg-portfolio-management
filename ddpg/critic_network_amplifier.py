import torch as T
import torch.nn as nn
import torch.optim as optim
import os

# Critic / Q-value Network / Q
# evaluate state/action pairs
class CriticNetworkAmplifier(nn.Module):
    def __init__(self, learning_rate, n_actions, name):
        super(CriticNetworkAmplifier, self).__init__()
        self.name = name
        state_size = n_actions * 8 + 1
        self.relu = nn.ReLU()

        self.fc_state = nn.Linear(state_size, 64)
        self.lstm = nn.LSTM(64, 64)
        self.__init_lstm(self.lstm)
        self.fc_action = nn.Linear(n_actions, 64)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(128, 64)
        self.bn = nn.LayerNorm(64)
        self.dropout = nn.Dropout(0.2)
        self.q = nn.Linear(64, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.device = T.device("cpu")
        self.to(self.device)

    def forward(self, state, action):
        state = self.relu(self.fc_state(state))
        state, _ = self.lstm(state.unsqueeze(0))
        state = self.relu(self.fc1(state.squeeze(0)))
        action = self.relu(self.fc_action(action))
        x = T.cat((state, action), dim=1)
        x = self.relu(self.fc2(x))
        x = self.bn(x)
        x = self.dropout(x)
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