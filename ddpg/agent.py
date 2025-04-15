import torch as T
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from .ou_action_noise import OUActionNoise
from .replay_buffer import ReplayBuffer
# from .actor_network_fc import ActorNetwork
# from .actor_network_lstm import ActorNetwork
from .actor_network_lstm_with_dropout import ActorNetwork
# from .critic_network_fc import CriticNetwork
# from .critic_network_lstm import CriticNetwork
from .critic_network_lstm_with_dropout import CriticNetwork
import os

# alpha and beta are the learning rate for actor and critic network, gamma is the discount factor for future reward
# tau is the "update rate" of the target networks oarameters (param_target = tau * param_online + (1-tau) * param_target)
class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, gamma, n_actions, max_size=300000, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size

        # self.actor = ActorNetwork(learning_rate=alpha, n_actions=n_actions, 
        #                           fc1_dims=256, fc2_dims=128, fc3_dims=64, name="actor")

        self.actor = ActorNetwork(learning_rate=alpha, n_actions=n_actions, lstm_size=128, fc_size=84, name="actor")

        # self.critic = CriticNetwork(learning_rate=beta, n_actions=n_actions, 
        #                             fc1_dims=256, fc2_dims=128, fc3_dims=64, name="critic")

        self.critic = CriticNetwork(learning_rate=beta, n_actions=n_actions, lstm_size=128, fc_size=84, name="critic")

        # self.target_actor = ActorNetwork(learning_rate=alpha, n_actions=n_actions, 
        #                                  fc1_dims=256, fc2_dims=128, fc3_dims=64, name="target_actor")

        self.target_actor = ActorNetwork(learning_rate=alpha, n_actions=n_actions, lstm_size=128, fc_size=84, name="target_actor")

        # self.target_critic = CriticNetwork(learning_rate=beta, n_actions=n_actions, 
        #                                    fc1_dims=256, fc2_dims=128, fc3_dims=64, name="target_critic")

        self.target_critic = CriticNetwork(learning_rate=beta, n_actions=n_actions, lstm_size=128, fc_size=84, name="target_critic")

        self.noise = OUActionNoise(mu=np.zeros(n_actions), sigma=0.3, theta=0.2)

        self.softmax = nn.Softmax(dim=-1)

        self.n_actions = n_actions

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, is_training):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        self.actor.train()
        # print("mu:", mu)

        # Epsilon-greedy exploration using noise
        if (is_training == True):
            epsilon = np.random.rand()
            if (self.memory.mem_cntr < 5000):
                if (epsilon < 0.5):
                    mu += T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
            elif (self.memory.mem_cntr < 10000):
                if (epsilon < 0.25):
                    mu += T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
            else:
                if (epsilon < 0.1):
                    mu += T.tensor(self.noise(), dtype=T.float).to(self.actor.device)       

        mu = self.softmax(mu)                               # Ensure actions sum to 1      
        
        return mu.cpu().detach().numpy()   

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        # Does not begin learning until the replay buffer is filled with at least a batch size
        if (self.memory.mem_cntr < self.batch_size):
            return None, None
        
        T.backends.cudnn.enabled = False

        states, action, reward, new_states, done = self.memory.sample_buffer(self.batch_size)
        
        # Change them to numpy arrays and will be used in critic network
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_states = T.tensor(new_states, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        states = T.tensor(states, dtype=T.float).to(self.critic.device)

        # Calculate the target actions like the bellman equation in Q-learning
        # The targets we want to move towards
        target_actions = self.target_actor.forward(new_states)
        target_critic_value = self.target_critic.forward(new_states, target_actions)
        critic_value = self.critic.forward(states, action)
        
        target = []
        for i in range(self.batch_size):
            target.append(reward[i] + self.gamma * (1 - done[i]) * target_critic_value[i])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        # Calculation of the loss function for the critic network
        self.critic.optimizer.zero_grad()
        # critic_loss = F.mse_loss(critic_value, target)
        critic_loss = F.huber_loss(critic_value, target, delta=1)
        cl = critic_loss.cpu().detach().numpy()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Calculation of the loss function for the actor network
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(states)
        actor_loss = -self.critic.forward(states, mu)
        actor_loss = T.mean(actor_loss)
        al = actor_loss.cpu().detach().numpy()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters(tau=self.tau)

        return al, cl

    def update_network_parameters(self, tau):
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        # Iterate over the dictionaries and look for the keys in the dict and update the value from the network
        for name in critic_state_dict:
            critic_state_dict[name] =  tau * critic_state_dict[name].clone() + (1 - tau) * target_critic_dict[name].clone()
            
        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self, saving_dir):
        if not os.path.isdir(saving_dir): 
            os.makedirs(saving_dir)

        self.actor.save_checkpoint(saving_dir)
        self.target_actor.save_checkpoint(saving_dir)
        self.critic.save_checkpoint(saving_dir)
        self.target_critic.save_checkpoint(saving_dir)

    def load_models(self, loading_dir):
        self.actor.load_checkpoint(loading_dir)
        self.target_actor.load_checkpoint(loading_dir)
        self.critic.load_checkpoint(loading_dir)
        self.target_critic.load_checkpoint(loading_dir)
