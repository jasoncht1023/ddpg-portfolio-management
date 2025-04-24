import torch as T
import torch.nn as nn
import torch.optim as optim
import os

# Actor / Policy Network / mu
# decide what to do based on the current state, outputs action values
class ActorNetworkAmplifier(nn.Module):   
    def __init__(self, learning_rate, n_actions, lstm_size, fc_size, name):
        super(ActorNetworkAmplifier, self).__init__()
        self.name = name
        input_size = n_actions * 8 + 1
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_size, num_layers=2, dropout=0.2)
        self.__init_lstm(self.lstm)

        self.fc = nn.Linear(lstm_size, fc_size)
        self.bn = nn.LayerNorm(fc_size)
        f1 = 0.003          
        nn.init.uniform_(self.fc.weight.data, -f1, f1)
        nn.init.uniform_(self.fc.bias.data, -f1, f1)

        self.mu = nn.Linear(fc_size, n_actions)

        self.softmax = nn.Softmax(dim=-1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.device = T.device("cpu")
        self.to(self.device)

    def forward(self, x):
        x = x.unsqueeze(0)
        output, (final_hidden_state, final_cell_state) = self.lstm(x)
        x = final_hidden_state[-1]
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.mu(x)
        x = T.tanh(x)
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