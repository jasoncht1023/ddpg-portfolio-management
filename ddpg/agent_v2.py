import torch as T
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from .ou_action_noise import OUActionNoise
from .replay_buffer import ReplayBuffer
from .actor_network_v2 import ActorNetwork
from .critic_network_v2 import CriticNetwork
import os
import matplotlib.pyplot as plt 

# alpha and beta are the learning rate for actor and critic network, gamma is the discount factor for future reward
# tau is the "update rate" of the target networks oarameters (param_target = tau * param_online + (1-tau) * param_target)
class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, gamma, n_actions, max_size=100000, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size

        self.actor = ActorNetwork(learning_rate=alpha, n_actions=n_actions, 
                                 fc1_dims=256, fc2_dims=128, fc3_dims=64, name="actor", chkpt_dir="temp")

        self.critic = CriticNetwork(learning_rate=beta, n_actions=n_actions, 
                                    lstm_size=100, name="critic", chkpt_dir="temp")

        self.target_actor = ActorNetwork(learning_rate=alpha, n_actions=n_actions, 
                                         fc1_dims=256, fc2_dims=128, fc3_dims=64, name="target_actor", chkpt_dir="temp")

        self.target_critic = CriticNetwork(learning_rate=beta, n_actions=n_actions, 
                                          lstm_size=100, name="target_critic", chkpt_dir="temp")

        self.noise = OUActionNoise(mu=np.zeros(n_actions), sigma=0.3, theta=0.2)

        self.softmax = nn.Softmax(dim=-1)

        self.n_actions = n_actions

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, is_training):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        # observation = observation.clone().detach().requires_grad_(True).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        # print("mu:", mu)
        # Epsilon-greedy exploration using noise
        if (is_training == True):
            epsilon = np.random.rand()
            if (self.memory.mem_cntr < 3000):
                if (epsilon < 0.5):
                    mu += T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
            elif (self.memory.mem_cntr < 6000):
                if (epsilon < 0.25):
                    mu += T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
            else:
                if (epsilon < 0.1):
                    mu += T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        
        self.actor.train()
        # print("mu_prime after adding noise:", mu_prime)
        mu = self.softmax(mu)                       # Actions sum to 1
        # print("softmax:", mu_prime)       

        return mu.cpu().detach().numpy()   

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        # Does not begin learning until the replay buffer is filled with at least a batch size
        if self.memory.mem_cntr < self.batch_size:
            return 0, 0
        T.backends.cudnn.enabled = False
        states, action, reward, new_states, done = self.memory.sample_buffer(self.batch_size)
        # state, action, reward, new_state, done = self.memory.pop_buffer()
        
        # Change them to numpy arrays and will be used in critic network
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_states = T.tensor(new_states, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        states = T.tensor(states, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        # Calculate the target actions like the bellman equation in Q-learning

        # The targets we want to move towards
        # refactor needed
        # target_actions = [
        #     self.target_actor.forward(input_tensor) for input_tensor in new_states
        # ]
        # target_actions = T.stack(target_actions).to(self.critic.device)
        target_actions = self.target_actor.forward(new_states)
        target_critic_value = self.target_critic.forward(new_states, target_actions)
        critic_value = self.critic.forward(states, action)
        
        y_arr = []
        for i in range(self.batch_size):
            y_arr.append(reward[i] + self.gamma * done[i] * target_critic_value[i])
        y_arr = T.tensor(y_arr).to(self.critic.device)
        y_arr = y_arr.view(self.batch_size, 1)

        # Calculation of the loss function for the critic network
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(critic_value, y_arr)
        cl = critic_loss.cpu().detach().numpy()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Calculation of the loss function for the actor network
        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(states)
        self.actor.train()
        actor_loss = -self.critic.forward(states, mu)
        actor_loss = T.mean(actor_loss)
        al = actor_loss.cpu().detach().numpy()
        self.critic.train()
        actor_loss.backward()
        self.actor.optimizer.step()
        self.actor.eval()      
        self.critic.eval()

        self.update_network_parameters()

        return al, cl

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

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

        """
        #Verify that the copy assignment worked correctly
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(target_critic_params)
        actor_state_dict = dict(target_actor_params)
        print('\nActor Networks', tau)
        for name, param in self.actor.named_parameters():
            print(name, T.equal(param, actor_state_dict[name]))
        print('\nCritic Networks', tau)
        for name, param in self.critic.named_parameters():
            print(name, T.equal(param, critic_state_dict[name]))
        input()
        """

    def save_models(self):
        if not os.path.isdir("temp"): 
            os.makedirs("temp")

        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def check_actor_params(self):
        current_actor_params = self.actor.named_parameters()
        current_actor_dict = dict(current_actor_params)
        original_actor_dict = dict(self.original_actor.named_parameters())
        original_critic_dict = dict(self.original_critic.named_parameters())
        current_critic_params = self.critic.named_parameters()
        current_critic_dict = dict(current_critic_params)
        print("Checking Actor parameters")

        for param in current_actor_dict:
            print(param, T.equal(original_actor_dict[param], current_actor_dict[param]))
        print("Checking critic parameters")
        for param in current_critic_dict:
            print(param, T.equal(original_critic_dict[param], current_critic_dict[param]))
        input()
