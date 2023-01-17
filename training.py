import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
#pd.options.mode.chained_assignment = None  # default='warn'

"""
import torch.optim as optim
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt

import scipy.stats as ss

import scipy.signal as sci
import math
import scipy as sp

import itertools

from datetime import datetime, timedelta
from datetime import date

import re
import argparse
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################################### Functions ############################################################

def sample(dic):
    ind = np.random.choice(dic['trial'].unique())
    return (torch.FloatTensor(dic['states'][dic['trial'] == ind].to_numpy()).to(device), torch.FloatTensor(dic['actions'][dic['trial'] == ind].to_numpy()).to(device), 
            torch.FloatTensor(dic['new_states'][dic['trial'] == ind].to_numpy()).to(device), torch.FloatTensor(dic['rewards'][dic['trial'] == ind].to_numpy()).to(device), 
            torch.Tensor(dic['terminals'][dic['trial'] == ind].to_numpy()).to(device))

def get_trajectory(dic, ind):
    return (torch.FloatTensor(dic['states'][dic['trial'] == ind].to_numpy()).to(device), torch.FloatTensor(dic['actions'][dic['trial'] == ind].to_numpy()).to(device), 
            torch.FloatTensor(dic['new_states'][dic['trial'] == ind].to_numpy()).to(device), torch.FloatTensor(dic['rewards'][dic['trial'] == ind].to_numpy()).to(device), 
            torch.Tensor(dic['terminals'][dic['trial'] == ind].to_numpy()).to(device))

def load_data(filename):
	print("Loading Data")
	np.random.seed(0)
	################### Load Saved dataset ##################
	# Read Saved dataset
	
	df = pd.read_csv(filename[0], header = 0, \
				names = ['trial','states','actions','new_states','rewards','terminals'], usecols = [1,2,3,4,5,6], lineterminator = "\n")
	df = df.replace([r'\n', r'\[', r'\]'], '', regex=True) 

	states = pd.DataFrame.from_records(np.array(df['states'].str.split(','))).astype(float)
	actions = pd.DataFrame.from_records(np.array(df['actions'].str.split(','))).astype(float)
	new_states = pd.DataFrame.from_records(np.array(df['new_states'].str.split(','))).astype(float)
	trial = df['trial'].astype(int)

	#Train/Test split
	trial_ind = np.arange(1,trial.iloc[-1]+1)
	train_trial = np.random.choice(trial_ind, size=200, replace=False)  #distrib proba for each value, could be useful to weight more "important" trajectories
	test_trial = np.delete(trial_ind, train_trial-1)

	train_ind = trial.isin(train_trial)
	test_ind = trial.isin(test_trial)

	train_set = {'trial': trial[train_ind],
					'states': states[train_ind],
					'actions': actions[train_ind],
					'new_states': new_states[train_ind],
					'rewards': df['rewards'][train_ind],
					'terminals': df['terminals'][train_ind].astype(bool)}

	test_set = {'trial': trial[test_ind],
					'states': states[test_ind],
					'actions': actions[test_ind],
					'new_states': new_states[test_ind],
					'rewards': df['rewards'][test_ind],
					'terminals': df['terminals'][test_ind].astype(bool)}
	### If multiple files are passed ###
	if len(filename) > 1:
		for i, file in enumerate(filename):
			if i > 0:
				df = pd.read_csv(file, header = 0, \
						names = ['trial','states','actions','new_states','rewards','terminals'], usecols = [1,2,3,4,5,6], lineterminator = "\n")
				df = df.replace([r'\n', r'\[', r'\]'], '', regex=True) 
			
				states = pd.DataFrame.from_records(np.array(df['states'].str.split(','))).astype(float)
				actions = pd.DataFrame.from_records(np.array(df['actions'].str.split(','))).astype(float)
				new_states = pd.DataFrame.from_records(np.array(df['new_states'].str.split(','))).astype(float)
				trial = df['trial'].astype(int)
				#Train/Test split
				trial_ind = np.arange(1,trial.iloc[-1]+1)
				train_trial = np.random.choice(trial_ind, size=200, replace=False)  #distrib proba for each value, could be useful to weight more "important" trajectories
				test_trial = np.delete(trial_ind, train_trial-1)

				train_ind = trial.isin(train_trial)
				test_ind = trial.isin(test_trial)

				train_set['trial'] = pd.concat([train_set['trial'],trial[train_ind]+ (i)*250 ], axis=0)
				train_set['states'] = pd.concat([train_set['states'], states[train_ind]], axis=0)
				train_set['actions'] = pd.concat([train_set['actions'], actions[train_ind]], axis=0)
				train_set['new_states'] = pd.concat([train_set['new_states'], new_states[train_ind]], axis=0)
				train_set['rewards'] = pd.concat([train_set['rewards'], df['rewards'][train_ind]], axis=0)
				train_set['terminals'] = pd.concat([train_set['terminals'], df['terminals'][train_ind]], axis=0)	

				test_set['trial'] = pd.concat([test_set['trial'], trial[test_ind]+ (i)*250 ], axis=0)
				test_set['states'] = pd.concat([test_set['states'], states[test_ind]], axis=0)
				test_set['actions'] = pd.concat([test_set['actions'], actions[test_ind]], axis=0)
				test_set['new_states'] = pd.concat([test_set['new_states'], new_states[test_ind]], axis=0)
				test_set['rewards'] = pd.concat([test_set['rewards'], df['rewards'][test_ind]], axis=0)
				test_set['terminals'] = pd.concat([test_set['terminals'], df['terminals'][test_ind]], axis=0)
	print("Train set number of trajectories: ", train_set['trial'].unique().shape[0], "Test set number of trajectories: ",test_set['trial'].unique().shape[0])
	return train_set, test_set
################################################## TD3 Agent ##################################################################

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)	#nn.Linear(256, 512)
		self.l3 = nn.Linear(256, action_dim)	#nn.Linear(512, 256)
		#self.l4 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		#a = F.relu(self.l3(a))
		return self.max_action * torch.tanh(self.l3(a))	#torch.tanh(self.l4(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256,256)	#nn.Linear(256, 512)
		self.l3 = nn.Linear(256, 1)	#nn.Linear(512, 256)
		#self.l4 = nn.Linear(256, 1)

		# Q2 architecture
		self.l5 = nn.Linear(state_dim + action_dim, 256)
		self.l6 = nn.Linear(256,256)	#nn.Linear(256, 512)
		self.l7 = nn.Linear(256, 1)	#nn.Linear(512, 256)
		#self.l8 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)#F.relu(self.l3(q1))
		#q1 = self.l4(q1)

		q2 = F.relu(self.l5(sa))
		q2 = F.relu(self.l6(q2))
		q2 = self.l7(q2)	#F.relu(self.l7(q2))
		#q2 = self.l8(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)	 #F.relu(self.l3(q1))
		#q1 = self.l4(q1)
		return q1


class TD3_BC(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		alpha=2.5,
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.alpha = alpha

		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state).to(device)	#.reshape(1, -1)
		return self.actor(state)	#.cpu().data.numpy().flatten()


	def train(self, trajectory, batch_size=4):
		self.total_it += 1
		a_loss = 0.0
		c_loss = 0.0
		# Sample replay buffer 
		states, actions, new_states, rewards, not_done = trajectory
		N_updates = states.size(dim=0)/batch_size
		for i in range(0, states.size(dim=0), batch_size):
			with torch.no_grad():
				# Select action according to policy and add clipped noise
				noise = (
					torch.randn_like(actions[i:i+batch_size][:]) * self.policy_noise
				).clamp(-self.noise_clip, self.noise_clip)
				
				next_action = (
					self.actor_target(new_states[i:i+batch_size][:]) + noise
				).clamp(-self.max_action, self.max_action)

				# Compute the target Q value
				target_Q1, target_Q2 = self.critic_target(new_states[i:i+batch_size][:], next_action)
				target_Q = torch.min(target_Q1, target_Q2)
				target_Q = rewards[i] + not_done[i] * self.discount * target_Q

			# Get current Q estimates
			#print(states[i:i+batch_size][:].shape, actions[i:i+batch_size][:].shape)
			current_Q1, current_Q2 = self.critic(states[i:i+batch_size][:], actions[i:i+batch_size][:])
			
			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			#if (self.total_it) % 5 == 0:
				#print("iteration ", self.total_it, " critic_loss: ", critic_loss)

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

		# Delayed policy updates
			if self.total_it % self.policy_freq == 0:

				# Compute actor loss
				pi = self.actor(states[i:i+batch_size][:])
				Q = self.critic.Q1(states[i:i+batch_size][:], pi)
				lmbda = self.alpha/Q.abs().mean().detach()

				actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, actions[i:i+batch_size][:])
				# Optimize the actor 
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()

				# Update the frozen target models
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data) 
				a_loss += actor_loss.detach().numpy()
			c_loss += critic_loss.detach().numpy()
		return c_loss/N_updates, a_loss/N_updates

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)

################################################## CQL Agent ###############################################################

'''
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=256, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)

        return action, log_prob
        
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        return action.detach().cpu()
    
    def get_det_action(self, state):
        mu, log_std = self.forward(state)
        return torch.tanh(mu).detach().cpu()

class DeepActor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, device, hidden_size=32, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DeepActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.device = device
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        in_dim = hidden_size+state_size

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(in_dim, hidden_size)
        self.fc3 = nn.Linear(in_dim, hidden_size)
        self.fc4 = nn.Linear(in_dim, hidden_size)


        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)
        #self.reset_parameters() # check if this improves training

    def reset_parameters(self, init_w=3e-3):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.mu.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.tensor):

        x = F.relu(self.fc1(state))
        x = torch.cat([x, state], dim=1)
        x = F.relu(self.fc2(x))
        x = torch.cat([x, state], dim=1)
        x = F.relu(self.fc3(x))
        x = torch.cat([x, state], dim=1)
        x = F.relu(self.fc4(x))  

        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)

        return action, log_prob

    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        return action.detach().cpu()
    
    def get_det_action(self, state):
        mu, log_std = self.forward(state)
        return torch.tanh(mu).detach().cpu()
    

class IQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256, seed=1, N=32, device="cuda:0"):
        super(IQN, self).__init__()
        torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.N = N  
        self.n_cos = 64
        self.layer_size = hidden_size
        self.pis = torch.FloatTensor([np.pi * i for i in range(1, self.n_cos + 1)]).view(1, 1, self.n_cos).to(device) # Starting from 0 as in the paper 
        self.device = device

        # Network Architecture
        self.head = nn.Linear(self.action_size + self.input_shape, hidden_size) 
        self.cos_embedding = nn.Linear(self.n_cos, hidden_size)
        self.ff_1 = nn.Linear(hidden_size, hidden_size)
        self.ff_2 = nn.Linear(hidden_size, 1)    

    def calc_input_layer(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        x = self.head(x)
        return x.flatten().shape[0]
        
    def calc_cos(self, batch_size, n_tau=32):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, n_tau).unsqueeze(-1).to(self.device)
        cos = torch.cos(taus * self.pis)

        assert cos.shape == (batch_size,n_tau,self.n_cos), "cos shape is incorrect"
        return cos, taus
    
    def forward(self, input, action, num_tau=32):
        """
        Quantile Calculation depending on the number of tau
        
        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]
        
        """
        batch_size = input.shape[0]

        x = torch.cat((input, action), dim=1)
        x = torch.relu(self.head(x  ))
        
        cos, taus = self.calc_cos(batch_size, num_tau) # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size*num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.layer_size) # (batch, n_tau, layer)
        
        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1) * cos_x).view(batch_size * num_tau, self.layer_size)  #batch_size*num_tau, self.cos_layer_out
        # Following reshape and transpose is done to bring the action in the same shape as batch*tau:
        # first 32 entries are tau for each action -> thats why each action one needs to be repeated 32 times 
        # x = [[tau1   action = [[a1
        #       tau1              a1   
        #        ..               ..
        #       tau2              a2
        #       tau2              a2
        #       ..]]              ..]]  
        #action = action.repeat(num_tau,1).reshape(num_tau,batch_size*self.action_size).transpose(0,1).reshape(batch_size*num_tau,self.action_size)
        #x = torch.cat((x,action),dim=1)
        x = torch.relu(self.ff_1(x))

        out = self.ff_2(x)
        
        return out.view(batch_size, num_tau, 1), taus
    
    def get_qvalues(self, inputs, action):
        quantiles, _ = self.forward(inputs, action, self.N)
        actions = quantiles.mean(dim=1)
        return actions  

class DeepIQN(nn.Module):
    def __init__(self, state_size, action_size, layer_size, seed, N, device="cuda:0"):
        super(DeepIQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.input_dim = action_size+state_size+layer_size
        self.N = N  
        self.n_cos = 64
        self.layer_size = layer_size
        self.pis = torch.FloatTensor([np.pi*i for i in range(1,self.n_cos+1)]).view(1,1,self.n_cos).to(device) # Starting from 0 as in the paper 
        self.device = device

        # Network Architecture

        self.head = nn.Linear(self.action_size+self.input_shape, layer_size) 
        self.ff_1 = nn.Linear(self.input_dim, layer_size)
        self.ff_2 = nn.Linear(self.input_dim, layer_size)
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        self.ff_3 = nn.Linear(self.input_dim, layer_size)
        self.ff_4 = nn.Linear(self.layer_size, 1)    
        #weight_init([self.head_1, self.ff_1])  

    def calc_input_layer(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        x = self.head(x)
        return x.flatten().shape[0]
        
    def calc_cos(self, batch_size, n_tau=32):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, n_tau).unsqueeze(-1).to(self.device) #(batch_size, n_tau, 1)  .to(self.device)
        cos = torch.cos(taus*self.pis)

        assert cos.shape == (batch_size,n_tau,self.n_cos), "cos shape is incorrect"
        return cos, taus
    
    def forward(self, input, action, num_tau=32):
        """
        Quantile Calculation depending on the number of tau
        
        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]
        
        """
        batch_size = input.shape[0]
        xs = torch.cat((input, action), dim=1)
        x = torch.relu(self.head(xs))
        x = torch.cat((x, xs), dim=1)
        x = torch.relu(self.ff_1(x))   
        x = torch.cat((x, xs), dim=1)
        x = torch.relu(self.ff_2(x))

        cos, taus = self.calc_cos(batch_size, num_tau) # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size*num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.layer_size) # (batch, n_tau, layer)
        
        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1)*cos_x).view(batch_size*num_tau, self.layer_size)  #batch_size*num_tau, self.cos_layer_out
        # Following reshape and transpose is done to bring the action in the same shape as batch*tau:
        # first 32 entries are tau for each action -> thats why each action one needs to be repeated 32 times 
        # x = [[tau1   action = [[a1
        #       tau1              a1   
        #        ..               ..
        #       tau2              a2
        #       tau2              a2
        #       ..]]              ..]]  
        action = action.repeat(num_tau,1).reshape(num_tau,batch_size*self.action_size).transpose(0,1).reshape(batch_size*num_tau,self.action_size)
        state = input.repeat(num_tau,1).reshape(num_tau,batch_size*self.input_shape).transpose(0,1).reshape(batch_size*num_tau,self.input_shape)
        
        x = torch.cat((x,action,state),dim=1)
        x = torch.relu(self.ff_3(x))

        out = self.ff_4(x)
        
        return out.view(batch_size, num_tau, 1), taus
    
    def get_qvalues(self, inputs, action):
        quantiles, _ = self.forward(inputs, action, self.N)
        actions = quantiles.mean(dim=1)
        return actions  



class CQLSAC(nn.Module):
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                        state_size,
                        action_size,
                        device
                ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(CQLSAC, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.device = device
        
        self.gamma = 0.99
        self.tau = 1e-2
        hidden_size = 256
        learning_rate = 5e-4
        self.clip_grad_param = 1

        self.target_entropy = -action_size  # -dim(A)

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate) 
        
        # CQL params
        self.with_lagrange = False
        self.temp = 1.0
        self.cql_weight = 1.0
        self.target_action_gap = 0.0
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=learning_rate) 
        
        # Actor Network 

        self.actor_local = Actor(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)     
        
        # Critic Network (w/ Target Network)

        self.critic1 = IQN(state_size, action_size, hidden_size, seed=1).to(device)
        self.critic2 = IQN(state_size, action_size, hidden_size, seed=2).to(device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = IQN(state_size, action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = IQN(state_size, action_size, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate) 

    def get_action(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        
        with torch.no_grad():
            if eval:
                action = self.actor_local.get_det_action(state)
            else:
                action = self.actor_local.get_action(state)
        return action.numpy()

    def calc_policy_loss(self, states, alpha):
        actions_pred, log_pis = self.actor_local.evaluate(states)

        q1 = self.critic1.get_qvalues(states, actions_pred.squeeze(0))   
        q2 = self.critic2.get_qvalues(states, actions_pred.squeeze(0))
        min_Q = torch.min(q1,q2).cpu()
        actor_loss = ((alpha * log_pis.cpu() - min_Q )).mean()
        return actor_loss, log_pis

    def _compute_policy_values(self, state_pi, state_q):
        with torch.no_grad():
            actions_pred, log_pis = self.actor_local.evaluate(state_pi)
        
        qs1 = self.critic1.get_qvalues(state_q, actions_pred)
        qs2 = self.critic2.get_qvalues(state_q, actions_pred)
        
        return qs1-log_pis, qs2-log_pis
    
    def _compute_random_values(self, state, actions, critic):
        random_values = critic.get_qvalues(state, actions)
        random_log_prstate = math.log(0.5 ** self.action_size)
        return random_values - random_log_prstate
    
    def train(self, step, experiences, gamma, d=1):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Compute alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            next_action, _ = self.actor_local.evaluate(next_states)
            #next_action = next_action.unsqueeze(1).repeat(1, 10, 1).view(next_action.shape[0] * 10, next_action.shape[1])
            #temp_next_states = next_states.unsqueeze(1).repeat(1, 10, 1).view(next_states.shape[0] * 10, next_states.shape[1])
            
            Q_target1_next, _ = self.critic1_target(next_states, next_action) #.view(states.shape[0], 10, 1).max(1)[0].view(-1, 1)
            # batch_size, num_tau, 1    
            Q_target2_next, _ = self.critic2_target(next_states, next_action) #.view(states.shape[0], 10, 1).max(1)[0].view(-1, 1)
            Q_target_next = torch.min(Q_target1_next, Q_target2_next).transpose(1,2)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards.cpu().unsqueeze(-1) + (gamma * (1 - dones.cpu().unsqueeze(-1)) * Q_target_next.cpu()) 


        # Compute critic loss
        q1, taus1 = self.critic1(states, actions)
        q2, taus2 = self.critic2(states, actions)
        assert Q_targets.shape == (256, 1, 32), "have shape: {}".format(Q_targets.shape)
        assert q1.shape == (256, 32, 1)
        
        # Quantile Huber loss
        td_error1 = Q_targets - q1.cpu()
        td_error2 = Q_targets - q2.cpu()
        
        assert td_error1.shape == (256, 32, 32), "wrong td error shape"
        huber_l_1 = calculate_huber_loss(td_error1, 1.0)
        huber_l_2 = calculate_huber_loss(td_error2, 1.0)
        
        quantil_l_1 = abs(taus1.cpu() - (td_error1.detach() < 0).float()) * huber_l_1 / 1.0
        quantil_l_2 = abs(taus2.cpu() - (td_error2.detach() < 0).float()) * huber_l_2 / 1.0

        critic1_loss = quantil_l_1.sum(dim=1).mean(dim=1).mean()
        critic2_loss = quantil_l_2.sum(dim=1).mean(dim=1).mean()

        
        # CQL addon

        random_actions = torch.FloatTensor(q1.shape[0] * 10, actions.shape[-1]).uniform_(-1, 1).to(self.device)
        num_repeat = int (random_actions.shape[0] / states.shape[0])
        temp_states = states.unsqueeze(1).repeat(1, num_repeat, 1).view(states.shape[0] * num_repeat, states.shape[1])
        temp_next_states = next_states.unsqueeze(1).repeat(1, num_repeat, 1).view(next_states.shape[0] * num_repeat, next_states.shape[1])
        
        current_pi_values1, current_pi_values2  = self._compute_policy_values(temp_states, temp_states)
        next_pi_values1, next_pi_values2 = self._compute_policy_values(temp_next_states, temp_states)
        
        random_values1 = self._compute_random_values(temp_states, random_actions, self.critic1).reshape(states.shape[0], num_repeat, 1)
        random_values2 = self._compute_random_values(temp_states, random_actions, self.critic2).reshape(states.shape[0], num_repeat, 1)

        current_pi_values1 = current_pi_values1.reshape(states.shape[0], num_repeat, 1)
        current_pi_values2 = current_pi_values2.reshape(states.shape[0], num_repeat, 1)
        next_pi_values1 = next_pi_values1.reshape(states.shape[0], num_repeat, 1)
        next_pi_values2 = next_pi_values2.reshape(states.shape[0], num_repeat, 1)      
        
        cat_q1 = torch.cat([random_values1, current_pi_values1, next_pi_values1], 1)
        cat_q2 = torch.cat([random_values2, current_pi_values2, next_pi_values2], 1)
        
        assert cat_q1.shape == (states.shape[0], 3 * num_repeat, 1), f"cat_q1 instead has shape: {cat_q1.shape}"
        assert cat_q2.shape == (states.shape[0], 3 * num_repeat, 1), f"cat_q2 instead has shape: {cat_q2.shape}"
        

        cql1_scaled_loss = (torch.logsumexp(cat_q1 / self.temp, dim=1).mean() * self.cql_weight * self.temp - q1.mean()) * self.cql_weight
        cql2_scaled_loss = (torch.logsumexp(cat_q2 / self.temp, dim=1).mean() * self.cql_weight * self.temp - q2.mean()) * self.cql_weight
        
        cql_alpha_loss = torch.FloatTensor([0.0])
        cql_alpha = torch.FloatTensor([0.0])
        if self.with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0).to(self.device)
            cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
            cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = (- cql1_scaled_loss - cql2_scaled_loss) * 0.5 
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()
        
        total_c1_loss = critic1_loss + cql1_scaled_loss
        total_c2_loss = critic2_loss + cql2_scaled_loss
        
        
        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        total_c1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        total_c2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        
        return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), cql1_scaled_loss.item(), cql2_scaled_loss.item(), current_alpha, cql_alpha_loss.item(), cql_alpha.item()

    def soft_update(self, local_model , target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], 32, 32), "huber loss has wrong shape"
    return loss

'''
######################### Train ####################################
def train(dataset, state_dim=24, action_dim=2, epochs=3000, train=True):  
	print("start Training")
	# Environment State Properties
	max_action = 1
	normalize = True
    # Agent parameters
	args = {
        # Experiment
        "policy": "TD3_BC",
        "seed": 0, 
        "eval_freq": 5e3,
        "max_timesteps": 250,   #1e6,
        "save_model": "store_true",
        "load_model": "",                 # Model load file name, "" doesn't load, "default" uses file_name
        # TD3
        "expl_noise": 0.1,
        "batch_size": 256,
        "discount": 0.99,
        "tau": 0.005,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "policy_freq": 2,
        # TD3 + BC
        "alpha": 2.5,
        "normalize": True,
        "state_dim": 14,
        "action_dim": 3,
        "max_action": 1,
        "discount": 0.99,
        "tau": 0.005,
        }

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args['discount'],
		"tau": args['tau'],
		# TD3
		"policy_noise": args['policy_noise'] * max_action,
		"noise_clip": args['noise_clip'] * max_action,
		"policy_freq": args['policy_freq'],
		# TD3 + BC
		"alpha": args['alpha']
	    }
    
	# Initialize Agent
	policy = TD3_BC(**kwargs) 
	## Load trained model ##
	print("---------------------------------------")
	print(f"Policy: {args['policy']}")	#, Seed: {args['seed']}")
	print("---------------------------------------")
	if train == True:
		batch_size = 256
		a_losses = []
		c_losses = []
		
		'''for trial in tqdm(dataset['trial'].unique()):
			trajectory = get_trajectory(dataset, trial)
			c_loss, a_loss = policy.train(trajectory, batch_size)	#, args, **kwargs
			if a_loss != 0.0:	#actor loss updated only once every "policy_freq" update, set to 0 otherwise
				a_losses.append(a_loss)
			c_losses.append(c_loss)
		'''
		for i in tqdm(range(1, epochs)):
			trajectory = sample(dataset)
			c_loss, a_loss = policy.train(trajectory, batch_size)	#, args, **kwargs
			if a_loss != 0.0:	#actor loss updated only once every "policy_freq" update, set to 0 otherwise
				a_losses.append(a_loss)
			c_losses.append(c_loss)
			
			if i%10 == 0:
				policy.save("models/TD3_BC_policy")
		_, actor_loss_curve = plt.subplots()
		actor_loss_curve.plot(a_losses)
		plt.savefig('training_plots/actor_losses_training_curve.png')

		_, critic_loss_curve = plt.subplots()
		critic_loss_curve.plot(c_losses)
		plt.savefig('training_plots/critic_losses_training_curve.png')
	else:
		print("load trained model")
		policy.load("models/TD3_BC_policy")
	
	return policy
######################### Evaluation ####################################
def evaluate(dataset, policy):
	print("start Evaluation")
	losses = []
	avg_loss = 0.0
	for i, trial in tqdm(enumerate(dataset['trial'].unique())):
		epoch_loss = 0.0
		states, actions, _,_,_ = get_trajectory(dataset, trial)	#new_states, rewards, not_done

		for j in range(states.size(dim=0)):

			#MSE on the action chosen by Agent on each state of a trajectory from the behaviour dataset
			pi_e = policy.select_action(states[j][:])
			pi_b = actions[j][:]
			epoch_loss += F.mse_loss(pi_e, pi_b)
			#MSE on final result of a trajectory?
			#if j==states.size(dim=0)-1:
				#print(pi_e, pi_b)
		epoch_loss = epoch_loss/states.size(dim=0)
		losses.append(epoch_loss.detach().numpy())
		#print("Loss for trajectory ", i, " (trial number: ", trial, ") = ", loss)

	_, evaluation_loss_curve = plt.subplots()
	evaluation_loss_curve.plot(losses)
	plt.savefig('training_plots/evaluation_losses.png')
	avg_loss = sum(losses)/len(losses)
	print("Average Evaluation loss: ", avg_loss)

########################## Main  ########################################

if __name__ == "__main__":
	filename = ["RL_dataset/Offline_reduced/AS_Offline_reduced.csv"]	#["RL_dataset/Offline_reduced/AAB_Offline_reduced.csv",
	 #"RL_dataset/Offline_reduced/AE_Offline_reduced.csv", "RL_dataset/Offline_reduced/AK_Offline_reduced.csv",
	 #"RL_dataset/Offline_reduced/AS_Offline_reduced.csv"]	#"RL_dataset/Offline_reduced/AAB_Offline_reduced.csv" #"RL_dataset/AAB.csv"
	train_set, test_set = load_data(filename)
	#state_dim=24, action_dim=2  
	#state dim must be multiple of 6
	policy = train(train_set,state_dim=14, action_dim=2, epochs=30000, train=True)	#set train=False to load pretrained model
	evaluate(test_set, policy)