import torch as T
import torch.nn as nn
import torch.optim as optim
import os

# Actor / Policy Network / mu
# decide what to do based on the current state, outputs action values
class ActorNetworkAmplifier(nn.Module):
    def __init__(self, learning_rate, n_actions, name):
        super(ActorNetworkAmplifier, self).__init__()
        self.name = name
        self.n_actions = n_actions
        self.input_size = n_actions * 8 + 1
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(self.input_size, 64)
        self.lstm = nn.LSTM(64, 64)
        self.__init_lstm(self.lstm)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.bn = nn.LayerNorm(64)
        self.dropout = nn.Dropout(0.4)
        self.fc4 = nn.Linear(64, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.softmax = nn.Softmax(dim=-1)
        # self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.device = T.device("cpu")
        self.to(self.device)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x, _ = self.lstm(x.unsqueeze(0))
        x = self.relu(self.fc2(x.squeeze(0)))
        x = self.relu(self.fc3(x))
        x = self.bn(x)
        x = self.dropout(x)
        x = T.tanh(self.fc4(x))
        x = x + 1
        return x
    
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

    def load_checkpoint(self, loading_dir, is_model_zipped):
        if (is_model_zipped == True):
            checkpoint_file = os.path.join(loading_dir, self.name + "_ddpg.zip")
        else:
            checkpoint_file = os.path.join(loading_dir, self.name + "_ddpg")
        if os.path.exists(checkpoint_file): 
            print("... loading checkpoint ...")
            self.load_state_dict(T.load(checkpoint_file))