import torch as T
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from .ou_action_noise import OUActionNoise
from .replay_buffer import ReplayBuffer
from .actor_network import ActorNetwork
from .critic_network import CriticNetwork
import os


# alpha and beta are the learning rate for actor and critic network, gamma is the discount factor for future reward
# tau is the "update rate" of the target networks oarameters (param_target = tau * param_online + (1-tau) * param_target)
class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, gamma, n_actions, max_size=500000, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.model_dir = "trained_model"

        self.actor = ActorNetwork(learning_rate=alpha, n_actions=n_actions, 
                                  input_dims=input_dims, fc_dims=400, name="actor", chkpt_dir=self.model_dir)

        self.critic = CriticNetwork(learning_rate=beta, n_actions=n_actions, 
                                    input_dims=input_dims, fc_dims=400, name="critic", chkpt_dir=self.model_dir)

        self.target_actor = ActorNetwork(learning_rate=alpha, n_actions=n_actions, 
                                         input_dims=input_dims, fc_dims=400, name="target_actor", chkpt_dir=self.model_dir)

        self.target_critic = CriticNetwork(learning_rate=beta, n_actions=n_actions, 
                                           input_dims=input_dims, fc_dims=400, name="target_critic", chkpt_dir=self.model_dir)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.softmax = nn.Softmax(dim=-1)

        self.n_actions = n_actions

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        # observation = observation.clone().detach().requires_grad_(True).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        # print("mu:", mu)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        # mu_prime = mu + T.tensor(np.random.normal(scale=0.05, size=self.n_actions)).to(self.actor.device)        # Adding gaussian noise
        self.actor.train()
        # print("mu_prime after adding noise:", mu_prime)
        mu_prime = self.softmax(mu_prime)                       # Actions sum to 1
        # print("softmax:", mu_prime)       
        mu_prime = mu_prime

        return mu_prime.cpu().detach().numpy()   

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        # Does not begin learning until the replay buffer is filled with at least a batch size
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        
        # Change them to numpy arrays and will be used in critic network
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        # Calculate the target actions like the bellman equation in Q-learning

        # The targets we want to move towards
        # refactor needed
        target_actions = [
            self.target_actor.forward(input_tensor) for input_tensor in new_state
        ]
        target_critic_value = [
            self.target_critic.forward(new_state[i], action)
            for i, action in enumerate(target_actions)
        ]
        critic_value = [
            self.critic.forward(state[i], action)
            for i, action in enumerate(target_actions)
        ]

        target_actions = T.stack(target_actions).to(self.critic.device)
        target_critic_value = T.stack(target_critic_value).to(self.critic.device)
        critic_value = T.stack(critic_value).to(self.critic.device)

        target = []
        for i in range(self.batch_size):
            target.append(reward[i] + self.gamma * (1 - done[i]) * target_critic_value[i])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        # Calculation of the loss function for the critic network
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(critic_value, target)
        critic_loss.backward()
        self.critic.optimizer.step()
        self.critic.eval()

        # Calculation of the loss function for the actor network
        self.actor.optimizer.zero_grad()
        mu = [self.actor.forward(input_tensor) for input_tensor in state]
        mu = T.stack(mu).to(self.actor.device)
        self.actor.train()
        actor_loss = [-self.critic.forward(state[i], action) for i, action in enumerate(mu)]
        actor_loss = T.stack(actor_loss).to(self.actor.device)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters(tau=self.tau)

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
        if not os.path.isdir(self.model_dir): 
            os.makedirs(self.model_dir)

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
