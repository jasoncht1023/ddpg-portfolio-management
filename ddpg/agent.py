import torch as T
import torch.nn.functional as F
import numpy as np
from .ou_action_noise import OUActionNoise
from .replay_buffer import ReplayBuffer
from .actor_network import ActorNetwork
from .critic_network import CriticNetwork


# alpha and beta are the learning rate for actor and critic network, gamma is the discount factor for future reward
class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, gamma=0.99, 
                 n_actions=2, max_size=1000000, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size

        # self.actor = ActorNetwork(alpha, input_dims, layer1_size,
        #                           layer2_size, n_actions=n_actions,
        #                           name='Actor')
        # self.critic = CriticNetwork(beta, input_dims, layer1_size,
        #                             layer2_size, n_actions=n_actions,
        #                             name='Critic')

        # self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
        #                                  layer2_size, n_actions=n_actions,
        #                                  name='TargetActor')
        # self.target_critic = CriticNetwork(beta, input_dims, layer1_size,
        #                                    layer2_size, n_actions=n_actions,
        #                                    name='TargetCritic')

        self.actor = ActorNetwork(learning_rate=alpha, n_actions=n_actions, name="Actor")

        self.critic = CriticNetwork(learning_rate=beta, n_actions=n_actions, name="Critic")

        self.target_actor = ActorNetwork(learning_rate=alpha, n_actions=n_actions, name="TargetActor")

        self.target_critic = CriticNetwork(learning_rate=beta, n_actions=n_actions, name="TargetCritic")

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        self.actor.train()
        # return mu_prime.cpu().detach().numpy()

        # For testing only, uncomment the above return statement and delete the below code
        mu_prime = mu_prime.cpu().detach().numpy()
        print("untransformed mu_prime:", mu_prime)
        mu_prime = np.array(list(map(abs, mu_prime)))
        print("mu_prime:", mu_prime)
        mu_prime = mu_prime / mu_prime.sum()            # actions sum to 1

        return mu_prime

    def remember(self, old_input_tensor, action, reward, new_input_tensor, done):
        self.memory.store_transition(old_input_tensor, action, reward, new_input_tensor, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        old_input_tensor, action, reward, new_input_tensor, done = self.memory.sample_buffer(self.batch_size)
        # change them to numpy arrays and will be used in critic network
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_input_tensor = T.tensor(new_input_tensor, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        old_input_tensor = T.tensor(old_input_tensor, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        # calculate the target actions like the bellman equation in Q-learning

        # refactor needed
        target_actions = [
            self.target_actor.forward(input_tensor) for input_tensor in old_input_tensor
        ]
        critic_value_ = [
            self.target_critic.forward(old_input_tensor[i], action)
            for i, action in enumerate(target_actions)
        ]
        critic_value = [
            self.critic.forward(old_input_tensor[i], action)
            for i, action in enumerate(target_actions)
        ]

        target_actions = T.stack(target_actions).to(self.critic.device)
        critic_value_ = T.stack(critic_value_).to(self.critic.device)
        critic_value = T.stack(critic_value).to(self.critic.device)

        y_arr = []
        for i in range(self.batch_size):
            y_arr.append(reward[i] + self.gamma * (1 - done[i]) * critic_value_[i])
        y_arr = T.tensor(y_arr).to(self.critic.device)
        y_arr = y_arr.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(critic_value, y_arr)
        critic_loss.backward()
        self.critic.optimizer.step()
        self.critic.eval()

        self.actor.optimizer.zero_grad()
        mu = [self.actor.forward(input_tensor) for input_tensor in old_input_tensor]
        mu = T.stack(mu).to(self.actor.device)
        self.actor.train()
        actor_loss = [
            -self.critic.forward(old_input_tensor[i], action)
            for i, action in enumerate(mu)
        ]
        actor_loss = T.stack(actor_loss).to(self.actor.device)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        self.actor.eval()

        self.update_network_parameters()

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

        for name in critic_state_dict:
            critic_state_dict[name] = (
                tau * critic_state_dict[name].clone()
                + (1 - tau) * target_critic_dict[name].clone()
            )

        # iterate over the dictionary and look for the keys in the dict and update the value from the network
        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = (
                tau * actor_state_dict[name].clone()
                + (1 - tau) * target_actor_dict[name].clone()
            )

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
