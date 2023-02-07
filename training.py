import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_
import copy
import math
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#pd.options.mode.chained_assignment = None  # default='warn'

"""
import matplotlib.pyplot as plt

import scipy.stats as ss

import scipy.signal as sci
import scipy as sp

import itertools

from datetime import datetime, timedelta
from datetime import date

import re
import argparse
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################################### Functions ############################################################
def update_cueballpos(prev_pos, velocity):
    dt = 0.0111
    new_pos = prev_pos + velocity*dt
    return new_pos

def update_cuepos(prev_pos_front, prev_pos_back, velocity):
    dt = 0.0111
    new_pos_front = prev_pos_front + velocity*dt
    new_pos_back = prev_pos_back + velocity*dt
    return new_pos_front, new_pos_back

def compute_reward_evaluation(cuevel, target_corner, reward):
    target_corner_angle = np.rad2deg(np.arctan2(target_corner[1], target_corner[0]))
    angle = np.rad2deg(np.arctan2(cuevel[1], cuevel[0]))
    if target_corner_angle < 90.0:
        angle = 180 - angle
    mean_pocket = 105.0736
    lbPocket = mean_pocket - 0.3106383 #= 104.7629617
    ubPocket = mean_pocket + 0.2543617 #= 105.3279617

    if reward == 100:
        reward = 100
    else:
        if angle <= ubPocket and angle >= lbPocket:     #Mathf.Max(currentTargetAngle, 
            reward = 100
            print("Funnel")
        else:
            reward = 0
    return reward


def weighted_sample(dic,prob_dist):
    ind = np.random.choice(dic['trial'][dic['terminals'] == True],p=prob_dist)
    return (torch.FloatTensor(dic['states'][dic['trial'] == ind].to_numpy()).to(device), torch.FloatTensor(dic['actions'][dic['trial'] == ind].to_numpy()).to(device), 
            torch.FloatTensor(dic['new_states'][dic['trial'] == ind].to_numpy()).to(device), torch.FloatTensor(dic['rewards'][dic['trial'] == ind].to_numpy()).to(device), 
            torch.from_numpy(dic['terminals'][dic['trial'] == ind].to_numpy(dtype=bool)).to(device))

def sample(dic):
    ind = np.random.choice(dic['trial'].unique())
    return (torch.FloatTensor(dic['states'][dic['trial'] == ind].to_numpy()).to(device), torch.FloatTensor(dic['actions'][dic['trial'] == ind].to_numpy()).to(device), 
            torch.FloatTensor(dic['new_states'][dic['trial'] == ind].to_numpy()).to(device), torch.FloatTensor(dic['rewards'][dic['trial'] == ind].to_numpy()).to(device), 
            torch.from_numpy(dic['terminals'][dic['trial'] == ind].to_numpy(dtype=bool)).to(device))

def get_trajectory(dic, ind):
    return (torch.FloatTensor(dic['states'][dic['trial'] == ind].to_numpy()).to(device), torch.FloatTensor(dic['actions'][dic['trial'] == ind].to_numpy()).to(device), 
            torch.FloatTensor(dic['new_states'][dic['trial'] == ind].to_numpy()).to(device), torch.FloatTensor(dic['rewards'][dic['trial'] == ind].to_numpy()).to(device), 
            torch.from_numpy(dic['terminals'][dic['trial'] == ind].to_numpy(dtype=bool)).to(device))

def normalize_states(self, eps = 1e-3):
		mean = self.states.mean(0,keepdims=True)
		std = self.states.std(0,keepdims=True) + eps
		self.states = (self.states - mean)/std
		self.new_states = (self.new_states - mean)/std
		return mean, std

def load_clean_data(filename):
	print("Loading Data")
	np.random.seed(0)
	################### Load Saved dataset ##################
	# Read Saved dataset
	
	df = pd.read_csv(filename[0], header = 0, \
				names = ['trial','states','actions','new_states','rewards','terminals'], usecols = [1,2,3,4,5,6], lineterminator = "\n")
	df = df.replace([r'\n', r'\[', r'\]', r'\r'], '', regex=True) 

	states = pd.DataFrame.from_records(np.array(df['states'].str.split(','))).astype(float)
	actions = pd.DataFrame.from_records(np.array(df['actions'].str.split(','))).astype(float)
	new_states = pd.DataFrame.from_records(np.array(df['new_states'].str.split(','))).astype(float)
	trial = df['trial'].astype(int)
	terminals = df['terminals'].astype(bool)
	#Train/Test split
	trial_clean = np.concatenate((trial.unique()[0:75],trial.unique()[224:]))
	trial_ind = np.arange(0,100)	#select only block 1,2.3 and 10 #1,len(trial.unique()) +1 if we want 250 index as well
	train_trial_ind = np.random.choice(trial_ind, size=int(0.8*len(trial_ind)), replace=False)  #distrib proba for each value, could be useful to weight more "important" trajectories
	
	test_trial_ind = np.delete(trial_ind, train_trial_ind)
	print(filename[0], " size: ", trial_clean[train_trial_ind].shape)
	train_trial = trial_clean[train_trial_ind]	
	test_trial = trial_clean[test_trial_ind]

	train_ind = trial.isin(train_trial)
	test_ind = trial.isin(test_trial)

	train_set = {'trial': trial[train_ind], 	#to have indexes from 1 to 50 and not 26 to 76 (complex when adding other files)
					'states': states[train_ind],
					'actions': actions[train_ind],
					'new_states': new_states[train_ind],
					'rewards': df['rewards'][train_ind],
					'terminals': terminals[train_ind]}

	test_set = {'trial': trial[test_ind],
					'states': states[test_ind],
					'actions': actions[test_ind],
					'new_states': new_states[test_ind],
					'rewards': df['rewards'][test_ind],
					'terminals': terminals[test_ind]}


	### If multiple files are passed ###
	if len(filename) > 1:
		for i, file in enumerate(filename):
			if i > 0:
				df = pd.read_csv(file, header = 0, \
						names = ['trial','states','actions','new_states','rewards','terminals'], usecols = [1,2,3,4,5,6], lineterminator = "\n")
				df = df.replace([r'\n', r'\[', r'\]', r'\r'], '', regex=True) 
			
				states = pd.DataFrame.from_records(np.array(df['states'].str.split(','))).astype(float)
				actions = pd.DataFrame.from_records(np.array(df['actions'].str.split(','))).astype(float)
				new_states = pd.DataFrame.from_records(np.array(df['new_states'].str.split(','))).astype(float)
				trial = df['trial'].astype(int)
				terminals = df['terminals'].astype(bool)
				#Train/Test split

				print(trial.unique()[0:75].shape,trial.unique()[224:])
				trial_clean = np.concatenate((trial.unique()[0:75],trial.unique()[224:]))
				trial_ind = np.arange(0,100)	#select only block 2 and 3 #1,len(trial.unique()) +1 if we want 250 index as well
				train_trial_ind = np.random.choice(trial_ind, size=int(0.8*len(trial_ind)), replace=False)  #distrib proba for each value, could be useful to weight more "important" trajectories
				
				#train_trial = trial_clean[train_trial_ind]
				print(file, trial_clean.shape, train_trial_ind.shape)
				print(" size in for loop: ", trial_clean[train_trial_ind].shape)
				test_trial_ind = np.delete(trial_ind, train_trial_ind)
				train_trial = trial_clean[train_trial_ind]
				test_trial = trial_clean[test_trial_ind]

				train_ind = trial.isin(train_trial)
				test_ind = trial.isin(test_trial)

				train_set['trial'] = pd.concat([train_set['trial'], trial[train_ind]+i*250], axis=0)
				train_set['states'] = pd.concat([train_set['states'], states[train_ind]], axis=0)
				train_set['actions'] = pd.concat([train_set['actions'], actions[train_ind]], axis=0)
				train_set['new_states'] = pd.concat([train_set['new_states'], new_states[train_ind]], axis=0)
				train_set['rewards'] = pd.concat([train_set['rewards'], df['rewards'][train_ind]], axis=0)
				train_set['terminals'] = pd.concat([train_set['terminals'], terminals[train_ind]], axis=0)	

				test_set['trial'] = pd.concat([test_set['trial'], trial[test_ind]+i*250], axis=0)
				test_set['states'] = pd.concat([test_set['states'], states[test_ind]], axis=0)
				test_set['actions'] = pd.concat([test_set['actions'], actions[test_ind]], axis=0)
				test_set['new_states'] = pd.concat([test_set['new_states'], new_states[test_ind]], axis=0)
				test_set['rewards'] = pd.concat([test_set['rewards'], df['rewards'][test_ind]], axis=0)
				test_set['terminals'] = pd.concat([test_set['terminals'], terminals[test_ind]], axis=0)
				
				#print(file,len(train_set['trial'].unique()), train_set['terminals'].value_counts())
	print("Train set number of trajectories: ", train_set['trial'].unique().shape[0], "Test set number of trajectories: ",test_set['trial'].unique().shape[0])
	return train_set, test_set


def load_data(filename):
	print("Loading Data")
	np.random.seed(0)
	################### Load Saved dataset ##################
	# Read Saved dataset
	
	df = pd.read_csv(filename[0], header = 0, \
				names = ['trial','states','actions','new_states','rewards','terminals'], usecols = [1,2,3,4,5,6], lineterminator = "\n")
	df = df.replace([r'\n', r'\[', r'\]', r'\r'], '', regex=True) 

	states = pd.DataFrame.from_records(np.array(df['states'].str.split(','))).astype(float)
	actions = pd.DataFrame.from_records(np.array(df['actions'].str.split(','))).astype(float)
	new_states = pd.DataFrame.from_records(np.array(df['new_states'].str.split(','))).astype(float)
	trial = df['trial'].astype(int)
	terminals = df['terminals'].astype(bool)

	#Train/Test split
	trial_ind = np.arange(0,len(trial.unique()-1))	#+1 if we want 250 index as well
	train_trial_ind = np.random.choice(trial_ind, size=int(0.8*len(trial_ind)), replace=False)  #distrib proba for each value, could be useful to weight more "important" trajectories
	train_trial = trial.unique()[train_trial_ind]
	test_trial = np.delete(trial.unique(), train_trial_ind)

	train_ind = trial.isin(train_trial)
	test_ind = trial.isin(test_trial)

	train_set = {'trial': trial[train_ind],
					'states': states[train_ind],
					'actions': actions[train_ind],
					'new_states': new_states[train_ind],
					'rewards': df['rewards'][train_ind],
					'terminals': terminals[train_ind]}

	test_set = {'trial': trial[test_ind],
					'states': states[test_ind],
					'actions': actions[test_ind],
					'new_states': new_states[test_ind],
					'rewards': df['rewards'][test_ind],
					'terminals': terminals[test_ind]}
	
	#print(filename[0], len(train_set['trial'].unique()), train_set['terminals'].value_counts())


	### If multiple files are passed ###
	if len(filename) > 1:
		for i, file in enumerate(filename):
			if i > 0:
				df = pd.read_csv(file, header = 0, \
						names = ['trial','states','actions','new_states','rewards','terminals'], usecols = [1,2,3,4,5,6], lineterminator = "\n")
				df = df.replace([r'\n', r'\[', r'\]', r'\r'], '', regex=True) 
			
				states = pd.DataFrame.from_records(np.array(df['states'].str.split(','))).astype(float)
				actions = pd.DataFrame.from_records(np.array(df['actions'].str.split(','))).astype(float)
				new_states = pd.DataFrame.from_records(np.array(df['new_states'].str.split(','))).astype(float)
				trial = df['trial'].astype(int)
				terminals = df['terminals'].astype(bool)
				#Train/Test split
				trial_ind = np.arange(0,len(trial.unique()-1))	#+1 if we want 250 index as well
				train_trial_ind = np.random.choice(trial_ind, size=int(0.8*len(trial_ind)), replace=False)  #distrib proba for each value, could be useful to weight more "important" trajectories
				train_trial = trial.unique()[train_trial_ind]
				test_trial = np.delete(trial.unique(), train_trial_ind)

				train_ind = trial.isin(train_trial)
				test_ind = trial.isin(test_trial)
				

				train_set['trial'] = pd.concat([train_set['trial'],trial[train_ind]+ (i)*250 ], axis=0)
				train_set['states'] = pd.concat([train_set['states'], states[train_ind]], axis=0)
				train_set['actions'] = pd.concat([train_set['actions'], actions[train_ind]], axis=0)
				train_set['new_states'] = pd.concat([train_set['new_states'], new_states[train_ind]], axis=0)
				train_set['rewards'] = pd.concat([train_set['rewards'], df['rewards'][train_ind]], axis=0)
				train_set['terminals'] = pd.concat([train_set['terminals'], terminals[train_ind]], axis=0)	

				test_set['trial'] = pd.concat([test_set['trial'], trial[test_ind]+ (i)*250 ], axis=0)
				test_set['states'] = pd.concat([test_set['states'], states[test_ind]], axis=0)
				test_set['actions'] = pd.concat([test_set['actions'], actions[test_ind]], axis=0)
				test_set['new_states'] = pd.concat([test_set['new_states'], new_states[test_ind]], axis=0)
				test_set['rewards'] = pd.concat([test_set['rewards'], df['rewards'][test_ind]], axis=0)
				test_set['terminals'] = pd.concat([test_set['terminals'], terminals[test_ind]], axis=0)
				
				#print(file,len(train_set['trial'].unique()), train_set['terminals'].value_counts())
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
		self.actions_upper_bound = torch.tensor([135,1,1])
		self.actions_lower_bound = torch.tensor([45,-1,-1])
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		#a = F.relu(self.l3(a))
		a = torch.tanh(self.l3(a))
		rescaled_actions = self.actions_lower_bound + (self.actions_upper_bound - self.actions_lower_bound) * (a + 1) / 2
		return 	rescaled_actions #torch.tanh(self.l4(a))

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


	def train(self, trajectory, batch_size=1):
		self.total_it += 1
		a_loss = 0.0
		c_loss = 0.0
		# Sample replay buffer 
		states, actions, new_states, rewards, terminals = trajectory
		not_terminals = ~terminals#.reshape(-1,1)
		#rewards = rewards.reshape(-1,1)
        #not_done = np.invert(terminals)
		N_updates = states.size(dim=0)/batch_size
		for i in range(0, states.size(dim=0), batch_size):
			with torch.no_grad():
				# Select action according to policy and add clipped noise
				noise = (
					torch.randn_like(actions[i:i+batch_size][:]) * self.policy_noise
				).clamp(-self.noise_clip, self.noise_clip)
				next_action = (
					self.actor_target(new_states[i:i+batch_size][:]) + noise	#.reshape(-1,1)		#Warning when action shape is 1, broadcasting fucks up the addition
				).clamp(-self.max_action, self.max_action)
				# Compute the target Q value
				target_Q1, target_Q2 = self.critic_target(new_states[i:i+batch_size][:], next_action)
				target_Q = torch.min(target_Q1, target_Q2)
				target_Q = rewards[i:i+batch_size] + not_terminals[i:i+batch_size] * self.discount * target_Q.squeeze()
	
			# Get current Q estimates
			#print(states[i:i+batch_size][:].shape, actions[i:i+batch_size][:].shape)
			current_Q1, current_Q2 = self.critic(states[i:i+batch_size][:], actions[i:i+batch_size][:])	#.reshape(-1,1))#Warning when action shape is 1
			
			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1.squeeze(), target_Q.squeeze()) + F.mse_loss(current_Q2.squeeze(), target_Q.squeeze())

			#if (self.total_it) % 5 == 0:
				#print("iteration ", self.total_it, " critic_loss: ", critic_loss)

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Delayed policy updates
			if i % self.policy_freq == 0:   #self.total_it

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
		return c_loss/N_updates, a_loss/(N_updates/self.policy_freq)

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")	# "_clean" +
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
from collections import OrderedDict
from copy import deepcopy

from ml_collections import ConfigDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F

from .utils import prefix_metrics

########## Utility Functions ############
class FullyConnectedNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, arch='256-256', orthogonal_init=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init

        d = input_dim
        modules = []
        hidden_sizes = [int(h) for h in arch.split('-')]

        for hidden_size in hidden_sizes:
            fc = nn.Linear(d, hidden_size)
            if orthogonal_init:
                nn.init.orthogonal_(fc.weight, gain=np.sqrt(2))
                nn.init.constant_(fc.bias, 0.0)
            modules.append(fc)
            modules.append(nn.ReLU())
            d = hidden_size

        last_fc = nn.Linear(d, output_dim)
        if orthogonal_init:
            nn.init.orthogonal_(last_fc.weight, gain=1e-2)
        else:
            nn.init.xavier_uniform_(last_fc.weight, gain=1e-2)

        nn.init.constant_(last_fc.bias, 0.0)
        modules.append(last_fc)

        self.network = nn.Sequential(*modules)

    def forward(self, input_tensor):
        return self.network(input_tensor)

class TanhGaussianPolicy(nn.Module):

    def __init__(self, observation_dim, action_dim, arch='256-256',
                 log_std_multiplier=1.0, log_std_offset=-1.0,
                 orthogonal_init=False, no_tanh=False):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = FullyConnectedNetwork(
            observation_dim, 2 * action_dim, arch, orthogonal_init
        )
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(self, observations, actions):
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian.log_prob(mean, log_std, actions)

    def forward(self, observations, deterministic=False, repeat=None):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian(mean, log_std, deterministic)


class SamplerPolicy(object):

    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    def __call__(self, observations, deterministic=False):
        with torch.no_grad():
            observations = torch.tensor(
                observations, dtype=torch.float32, device=self.device
            )
            actions, _ = self.policy(observations, deterministic)
            actions = actions.cpu().numpy()
        return actions


class FullyConnectedQFunction(nn.Module):

    def __init__(self, observation_dim, action_dim, arch='256-256', orthogonal_init=False):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init
        self.network = FullyConnectedNetwork(
            observation_dim + action_dim, 1, arch, orthogonal_init
        )

    @multiple_action_q_function
    def forward(self, observations, actions):
        input_tensor = torch.cat([observations, actions], dim=-1)
        return torch.squeeze(self.network(input_tensor), dim=-1)


def soft_target_update(network, target_network, soft_target_update_rate):
    target_network_params = {k: v for k, v in target_network.named_parameters()}
    for k, v in network.named_parameters():
        target_network_params[k].data = (
            (1 - soft_target_update_rate) * target_network_params[k].data
            + soft_target_update_rate * v.data
        )


class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32)
        )

    def forward(self):
        return self.constant


def prefix_metrics(metrics, prefix):
    return {
        '{}/{}'.format(prefix, key): value for key, value in metrics.items()
    }
################## CQL class Definition ########################
class ConservativeSAC(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.discount = 0.99
        config.alpha_multiplier = 1.0
        config.use_automatic_entropy_tuning = True
        config.backup_entropy = False
        config.target_entropy = 0.0
        config.policy_lr = 3e-4
        config.qf_lr = 3e-4
        config.optimizer_type = 'adam'
        config.soft_target_update_rate = 5e-3
        config.target_update_period = 1
        config.use_cql = True
        config.cql_n_actions = 10
        config.cql_importance_sample = True
        config.cql_lagrange = False
        config.cql_target_action_gap = 1.0
        config.cql_temp = 1.0
        config.cql_min_q_weight = 5.0
        config.cql_max_target_backup = False
        config.cql_clip_diff_min = -np.inf
        config.cql_clip_diff_max = np.inf

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, policy, qf1, qf2, target_qf1, target_qf2):
        self.config = ConservativeSAC.get_default_config(config)
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2

        optimizer_class = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
        }[self.config.optimizer_type]

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(), self.config.policy_lr,
        )
        self.qf_optimizer = optimizer_class(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), self.config.qf_lr
        )

        if self.config.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = optimizer_class(
                self.log_alpha.parameters(),
                lr=self.config.policy_lr,
            )
        else:
            self.log_alpha = None

        if self.config.cql_lagrange:
            self.log_alpha_prime = Scalar(1.0)
            self.alpha_prime_optimizer = optimizer_class(
                self.log_alpha_prime.parameters(),
                lr=self.config.qf_lr,
            )

        self.update_target_network(1.0)
        self._total_steps = 0

    def update_target_network(self, soft_target_update_rate):
        soft_target_update(self.qf1, self.target_qf1, soft_target_update_rate)
        soft_target_update(self.qf2, self.target_qf2, soft_target_update_rate)

    def train(self, batch, bc=False):
        self._total_steps += 1

        observations = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_observations = batch['new_states']
        dones = batch['terminals']

        new_actions, log_pi = self.policy(observations)

        if self.config.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha() * (log_pi + self.config.target_entropy).detach()).mean()
            alpha = self.log_alpha().exp() * self.config.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.config.alpha_multiplier)

        """ Policy loss """
        if bc:
            log_probs = self.policy.log_prob(observations, actions)
            policy_loss = (alpha*log_pi - log_probs).mean()
        else:
            q_new_actions = torch.min(
                self.qf1(observations, new_actions),
                self.qf2(observations, new_actions),
            )
            policy_loss = (alpha*log_pi - q_new_actions).mean()

        """ Q function loss """
        q1_pred = self.qf1(observations, actions)
        q2_pred = self.qf2(observations, actions)

        if self.config.cql_max_target_backup:
            new_next_actions, next_log_pi = self.policy(next_observations, repeat=self.config.cql_n_actions)
            target_q_values, max_target_indices = torch.max(
                torch.min(
                    self.target_qf1(next_observations, new_next_actions),
                    self.target_qf2(next_observations, new_next_actions),
                ),
                dim=-1
            )
            next_log_pi = torch.gather(next_log_pi, -1, max_target_indices.unsqueeze(-1)).squeeze(-1)
        else:
            new_next_actions, next_log_pi = self.policy(next_observations)
            target_q_values = torch.min(
                self.target_qf1(next_observations, new_next_actions),
                self.target_qf2(next_observations, new_next_actions),
            )

        if self.config.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        td_target = rewards + (1. - dones) * self.config.discount * target_q_values
        qf1_loss = F.mse_loss(q1_pred, td_target.detach())
        qf2_loss = F.mse_loss(q2_pred, td_target.detach())


        ### CQL
        if not self.config.use_cql:
            qf_loss = qf1_loss + qf2_loss
        else:
            batch_size = actions.shape[0]
            action_dim = actions.shape[-1]
            cql_random_actions = actions.new_empty((batch_size, self.config.cql_n_actions, action_dim), requires_grad=False).uniform_(-1, 1)
            cql_current_actions, cql_current_log_pis = self.policy(observations, repeat=self.config.cql_n_actions)
            cql_next_actions, cql_next_log_pis = self.policy(next_observations, repeat=self.config.cql_n_actions)
            cql_current_actions, cql_current_log_pis = cql_current_actions.detach(), cql_current_log_pis.detach()
            cql_next_actions, cql_next_log_pis = cql_next_actions.detach(), cql_next_log_pis.detach()

            cql_q1_rand = self.qf1(observations, cql_random_actions)
            cql_q2_rand = self.qf2(observations, cql_random_actions)
            cql_q1_current_actions = self.qf1(observations, cql_current_actions)
            cql_q2_current_actions = self.qf2(observations, cql_current_actions)
            cql_q1_next_actions = self.qf1(observations, cql_next_actions)
            cql_q2_next_actions = self.qf2(observations, cql_next_actions)

            cql_cat_q1 = torch.cat(
                [cql_q1_rand, torch.unsqueeze(q1_pred, 1), cql_q1_next_actions, cql_q1_current_actions], dim=1
            )
            cql_cat_q2 = torch.cat(
                [cql_q2_rand, torch.unsqueeze(q2_pred, 1), cql_q2_next_actions, cql_q2_current_actions], dim=1
            )
            cql_std_q1 = torch.std(cql_cat_q1, dim=1)
            cql_std_q2 = torch.std(cql_cat_q2, dim=1)

            if self.config.cql_importance_sample:
                random_density = np.log(0.5 ** action_dim)
                cql_cat_q1 = torch.cat(
                    [cql_q1_rand - random_density,
                     cql_q1_next_actions - cql_next_log_pis.detach(),
                     cql_q1_current_actions - cql_current_log_pis.detach()],
                    dim=1
                )
                cql_cat_q2 = torch.cat(
                    [cql_q2_rand - random_density,
                     cql_q2_next_actions - cql_next_log_pis.detach(),
                     cql_q2_current_actions - cql_current_log_pis.detach()],
                    dim=1
                )

            cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.config.cql_temp, dim=1) * self.config.cql_temp
            cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.config.cql_temp, dim=1) * self.config.cql_temp

            """Subtract the log likelihood of data"""
            cql_qf1_diff = torch.clamp(
                cql_qf1_ood - q1_pred,
                self.config.cql_clip_diff_min,
                self.config.cql_clip_diff_max,
            ).mean()
            cql_qf2_diff = torch.clamp(
                cql_qf2_ood - q2_pred,
                self.config.cql_clip_diff_min,
                self.config.cql_clip_diff_max,
            ).mean()

            if self.config.cql_lagrange:
                alpha_prime = torch.clamp(torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0)
                cql_min_qf1_loss = alpha_prime * self.config.cql_min_q_weight * (cql_qf1_diff - self.config.cql_target_action_gap)
                cql_min_qf2_loss = alpha_prime * self.config.cql_min_q_weight * (cql_qf2_diff - self.config.cql_target_action_gap)

                self.alpha_prime_optimizer.zero_grad()
                alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss)*0.5
                alpha_prime_loss.backward(retain_graph=True)
                self.alpha_prime_optimizer.step()
            else:
                cql_min_qf1_loss = cql_qf1_diff * self.config.cql_min_q_weight
                cql_min_qf2_loss = cql_qf2_diff * self.config.cql_min_q_weight
                alpha_prime_loss = observations.new_tensor(0.0)
                alpha_prime = observations.new_tensor(0.0)


            qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss


        if self.config.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        if self.total_steps % self.config.target_update_period == 0:
            self.update_target_network(
                self.config.soft_target_update_rate
            )


        metrics = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            qf1_loss=qf1_loss.item(),
            qf2_loss=qf2_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
            average_qf1=q1_pred.mean().item(),
            average_qf2=q2_pred.mean().item(),
            average_target_q=target_q_values.mean().item(),
            total_steps=self.total_steps,
        )

        if self.config.use_cql:
            metrics.update(prefix_metrics(dict(
                cql_std_q1=cql_std_q1.mean().item(),
                cql_std_q2=cql_std_q2.mean().item(),
                cql_q1_rand=cql_q1_rand.mean().item(),
                cql_q2_rand=cql_q2_rand.mean().item(),
                cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
                cql_qf1_diff=cql_qf1_diff.mean().item(),
                cql_qf2_diff=cql_qf2_diff.mean().item(),
                cql_q1_current_actions=cql_q1_current_actions.mean().item(),
                cql_q2_current_actions=cql_q2_current_actions.mean().item(),
                cql_q1_next_actions=cql_q1_next_actions.mean().item(),
                cql_q2_next_actions=cql_q2_next_actions.mean().item(),
                alpha_prime_loss=alpha_prime_loss.item(),
                alpha_prime=alpha_prime.item(),
            ), 'cql'))

        return metrics

    def torch_to_device(self, device):
        for module in self.modules:
            module.to(device)

    @property
    def modules(self):
        modules = [self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2]
        if self.config.use_automatic_entropy_tuning:
            modules.append(self.log_alpha)
        if self.config.cql_lagrange:
            modules.append(self.log_alpha_prime)
        return modules

    @property
    def total_steps(self):
        return self._total_steps

################## Flags #############################
FLAGS_DEF = define_flags_with_default(
    env='halfcheetah-medium-v2',
    max_traj_length=1000,
    seed=42,
    device='cuda',
    save_model=False,
    batch_size=256,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.999,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    n_epochs=2000,
    bc_epochs=0,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=5,

    cql=ConservativeSAC.get_default_config(),
    logging=WandBLogger.get_default_config(),
)

################## Policy Definition ###########################

def CQLSAC():
	FLAGS = absl.flags.FLAGS
	eval_sampler = dataset
	policy = TanhGaussianPolicy(
		eval_sampler.env.observation_space.shape[0],
		eval_sampler.env.action_space.shape[0],
		arch=FLAGS.policy_arch,
		log_std_multiplier=FLAGS.policy_log_std_multiplier,
		log_std_offset=FLAGS.policy_log_std_offset,
		orthogonal_init=FLAGS.orthogonal_init,
	)

	qf1 = FullyConnectedQFunction(
		eval_sampler.env.observation_space.shape[0],
		eval_sampler.env.action_space.shape[0],
		arch=FLAGS.qf_arch,
		orthogonal_init=FLAGS.orthogonal_init,
	)
	target_qf1 = deepcopy(qf1)

	qf2 = FullyConnectedQFunction(
		eval_sampler.env.observation_space.shape[0],
		eval_sampler.env.action_space.shape[0],
		arch=FLAGS.qf_arch,
		orthogonal_init=FLAGS.orthogonal_init,
	)
	target_qf2 = deepcopy(qf2)

	if FLAGS.cql.target_entropy >= 0.0:
		FLAGS.cql.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

	sac = ConservativeSAC(FLAGS.cql, policy, qf1, qf2, target_qf1, target_qf2)
	sac.torch_to_device(FLAGS.device)

	sampler_policy = SamplerPolicy(policy, FLAGS.device)

'''
#################################################################################################################################
'''
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor_CQL(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor_CQL, self).__init__()
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


class Critic_CQL(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, seed=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(Critic_CQL, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size+action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class CQLSAC(nn.Module):
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                        state_size,
                        action_size,
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        self.actor_local = Actor_CQL(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)     
        
        # Critic Network (w/ Target Network)

        self.critic1 = Critic_CQL(state_size, action_size, hidden_size, 2).to(device)
        self.critic2 = Critic_CQL(state_size, action_size, hidden_size, 1).to(device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = Critic_CQL(state_size, action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic_CQL(state_size, action_size, hidden_size).to(device)
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

        q1 = self.critic1(states, actions_pred.squeeze(0))   
        q2 = self.critic2(states, actions_pred.squeeze(0))
        min_Q = torch.min(q1,q2).cpu()
        actor_loss = ((alpha * log_pis.cpu() - min_Q )).mean()
        return actor_loss, log_pis

    def _compute_policy_values(self, obs_pi, obs_q):
        #with torch.no_grad():
        actions_pred, log_pis = self.actor_local.evaluate(obs_pi)
        
        qs1 = self.critic1(obs_q, actions_pred)
        qs2 = self.critic2(obs_q, actions_pred)
        
        return qs1 - log_pis.detach(), qs2 - log_pis.detach()
    
    def _compute_random_values(self, obs, actions, critic):
        random_values = critic(obs, actions)
        random_log_probs = math.log(0.5 ** self.action_size)
        return random_values - random_log_probs
    
    def train(self, experiences, batch_size=1):
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
        not_dones = ~ dones
        actor_losses_ = []
        alpha_losses_ = []
        critic1_losses_ = []
        critic2_losses_ = []
        cql1_scaled_losses_ = []
        cql2_scaled_losses_ = []

        for i in range(0, states.size(dim=0), batch_size): 
            # ---------------------------- update actor ---------------------------- #
            current_alpha = copy.deepcopy(self.alpha)
            actor_loss, log_pis = self.calc_policy_loss(states[i:i+batch_size][:], current_alpha)
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
                next_action, new_log_pi = self.actor_local.evaluate(next_states[i:i+batch_size][:])
                Q_target1_next = self.critic1_target(next_states[i:i+batch_size][:], next_action)
                Q_target2_next = self.critic2_target(next_states[i:i+batch_size][:], next_action)
                Q_target_next = torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * new_log_pi
                # Compute Q targets for current states (y_i)
                Q_targets = rewards + (self.gamma * not_dones[i:i+batch_size][:] * Q_target_next) 


            # Compute critic loss
            q1 = self.critic1(states[i:i+batch_size][:], actions[i:i+batch_size][:])
            q2 = self.critic2(states[i:i+batch_size][:], actions[i:i+batch_size][:])

            critic1_loss = F.mse_loss(q1, Q_targets)
            critic2_loss = F.mse_loss(q2, Q_targets)
            
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
            

            cql1_scaled_loss = ((torch.logsumexp(cat_q1 / self.temp, dim=1).mean() * self.cql_weight * self.temp) - q1.mean()) * self.cql_weight
            cql2_scaled_loss = ((torch.logsumexp(cat_q2 / self.temp, dim=1).mean() * self.cql_weight * self.temp) - q2.mean()) * self.cql_weight
            
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
            
            ## store losses ##
            alpha_losses_.append(alpha_loss.item())
            critic1_losses_.append(critic1_loss.item())
            critic2_losses_.append(critic2_loss.item())
            cql1_scaled_losses_.append(cql1_scaled_loss.item())
            cql2_scaled_losses_.append(cql2_scaled_loss.item())
            actor_losses_.append(actor_loss.item())

            #actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), cql1_scaled_loss.item(), cql2_scaled_loss.item(), current_alpha, cql_alpha_loss.item(), cql_alpha.item()
            return alpha_losses_, critic1_losses_, critic2_losses_,cql1_scaled_losses_, cql2_scaled_losses_, actor_losses_

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
'''

'''
class Actor_CQL(nn.Module):
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
        super(Actor_CQL, self).__init__()
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
        log_prob = (dist.log_prob(e) - torch.log(1 - action[0].pow(2) + epsilon)).sum(1, keepdim=True)

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
                        action_size
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
	
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        self.actor_local = Actor_CQL(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)     
        
        # Critic Network (w/ Target Network)

        self.critic1 = IQN(state_size, action_size, hidden_size, seed=1, device = self.device).to(device)
        self.critic2 = IQN(state_size, action_size, hidden_size, seed=2, device = self.device).to(device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = IQN(state_size, action_size, hidden_size, device = self.device).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = IQN(state_size, action_size, hidden_size, device = self.device).to(device)
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
    
    def train(self, experiences, batch_size=1):	#step, gamma, d=1
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
        states, actions, next_states, rewards, dones = experiences
        not_dones = ~ dones
        actor_losses_ = []
        alpha_losses_ = []
        critic1_losses_ = []
        critic2_losses_ = []
        cql1_scaled_losses_ = []
        cql2_scaled_losses_ = []

        for i in range(0, states.size(dim=0), batch_size):   
            # ---------------------------- update actor ---------------------------- #
            current_alpha = copy.deepcopy(self.alpha)
            actor_loss, log_pis = self.calc_policy_loss(states[i:i+batch_size][:], current_alpha)
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
                next_action, _ = self.actor_local.evaluate(next_states[i:i+batch_size][:])
                #next_action = next_action.unsqueeze(1).repeat(1, 10, 1).view(next_action.shape[0] * 10, next_action.shape[1])
                #temp_next_states = next_states.unsqueeze(1).repeat(1, 10, 1).view(next_states.shape[0] * 10, next_states.shape[1])
                
                Q_target1_next, _ = self.critic1_target(next_states[i:i+batch_size][:], next_action) #.view(states.shape[0], 10, 1).max(1)[0].view(-1, 1)
                # batch_size, num_tau, 1    
                Q_target2_next, _ = self.critic2_target(next_states[i:i+batch_size][:], next_action) #.view(states.shape[0], 10, 1).max(1)[0].view(-1, 1)
                Q_target_next = torch.min(Q_target1_next, Q_target2_next).transpose(1,2)

                print("target shape: ", Q_target_next.cpu().shape, not_dones[i:i+batch_size].cpu().unsqueeze(-1).shape, not_dones[i:i+batch_size].cpu().shape)
                # Compute Q targets for current states (y_i)
                Q_targets = rewards.cpu().unsqueeze(-1) + (self.gamma * not_dones[i:i+batch_size].cpu() * Q_target_next.cpu()) 


            # Compute critic loss
            q1, taus1 = self.critic1(states[i:i+batch_size][:], actions[i:i+batch_size][:])
            q2, taus2 = self.critic2(states[i:i+batch_size][:], actions[i:i+batch_size][:])
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
		
            alpha_losses_.append(alpha_loss.item())
            critic1_losses_.append(critic1_loss.item())
            critic2_losses_.append(critic2_loss.item())
            cql1_scaled_losses_.append(cql1_scaled_loss.item())
            cql2_scaled_losses_.append(cql2_scaled_loss.item())
            actor_losses_.append(actor_loss.item())

        return actor_losses_, alpha_losses_, critic1_losses_, critic2_losses_, cql1_scaled_losses_, cql2_scaled_losses_#actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), cql1_scaled_loss.item(), cql2_scaled_loss.item(), current_alpha, cql_alpha_loss.item(), cql_alpha.item()

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
def train(dataset, test_set, state_dim=14, action_dim=2, epochs=3000, train=True, model = "TD3_BC"):  
	print("start Training")
	# Environment State Properties
	max_action = 1
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
        "expl_noise": 0.3,
        "batch_size": 256,
        "discount": 0.99,
        "tau": 0.005,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "policy_freq": 2,
        # TD3 + BC
        "alpha": 3.,#2.5
        "normalize": True,
        "state_dim": 14,
        "action_dim": 3,
        "max_action": 10,	#different max_action for each dimension of the Action space?
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

	if model == "CQL_SAC":
		# Initialize Agent
		policy = CQLSAC(state_dim,action_dim)
		print("---------------------------------------")
		print(f"Policy: {args['policy']}")
		print("---------------------------------------")
		if train == True:
			alpha_losses = []
			critic1_losses = []
			critic2_losses = []
			policy_losses = []

			for i in tqdm(range(1, epochs)):
				trajectory = sample(dataset)	#weighted_sample(dataset, prob_dist)
				#trajectory = sample(dataset)
				actor_loss, alpha_loss, critic1_loss, critic2_loss, cql1_scaled_loss, cql2_scaled_loss= policy.train(trajectory, batch_size = 32)  #, current_alpha, cql_alpha_loss, cql_alpha 
				print("alpha loss: ", alpha_loss)
				alpha_losses.append(alpha_loss)
				critic1_losses.append(critic1_loss + cql1_scaled_loss)
				critic2_losses.append(critic2_loss + cql2_scaled_loss)
				policy_losses.append(actor_loss)
				
				#if i%10 == 0:
					#policy.save("models/CQL_SAC_policy")
			_, alpha_loss_curve = plt.subplots()
			alpha_loss_curve.plot(alpha_losses)
			plt.savefig('training_plots/CQL_SAC/alpha_losses_training_curve.png')

			_, critic1_loss_curve = plt.subplots()
			critic1_loss_curve.plot(critic1_losses)
			plt.savefig('training_plots/CQL_SAC/critic1_losses_training_curve.png')

			_, critic2_loss_curve = plt.subplots()
			critic2_loss_curve.plot(alpha_losses)
			plt.savefig('training_plots/CQL_SAC/critic2_losses_training_curve.png')

			_, policy_loss_curve = plt.subplots()
			policy_loss_curve.plot(policy_losses)
			plt.savefig('training_plots/CQL_SAC/policy_losses_training_curve.png')
	
	elif model == "TD3_BC":
		# Initialize Agent
		policy = TD3_BC(**kwargs) 
		print("---------------------------------------")
		print(f"Policy: {args['policy']}")	#, Seed: {args['seed']}")
		print("---------------------------------------")
		if train == True:
			a_losses = []
			c_losses = []
			evaluation_losses = []

			
			trajectory_rewards = dataset['rewards'][dataset['terminals']]

			prob_dist = np.ones(len(trajectory_rewards))	#len(dataset['trial'].unique())-10)#-1 Because last trial has no terminal state (to be changed in future)

			prob_dist[trajectory_rewards != -10.0] = 50	#500 times higher probs than other traj to be sampled
			prob_dist = prob_dist/prob_dist.sum()
			
			for i in tqdm(range(1, epochs)):
				trajectory = weighted_sample(dataset, prob_dist)    #sample(dataset) 	   

				#trajectory = sample(dataset)
				c_loss, a_loss = policy.train(trajectory, batch_size = 256)	#, args, **kwargs
				if a_loss != 0.0:	#actor loss updated only once every "policy_freq" update, set to 0 otherwise
					a_losses.append(a_loss)
				c_losses.append(c_loss)

				
				if i%2000 == 0:
					plot_animated_learnt_policy(test_set, policy, i)
				
				if i%1000 == 0:
					#evaluate_agent_performance(test_set, policy)
					evaluation_loss = evaluate_close_to_human_behaviour(test_set, policy,i)
					evaluation_losses = np.append(evaluation_losses, evaluation_loss.detach().numpy())
					policy.save("models/TD3_BC_policy")
			
			_, evaluation_loss_curve = plt.subplots()
			evaluation_loss_curve.plot(evaluation_losses)
			plt.savefig('training_plots/TD3_BC/evaluation_losses_during_training.png')

			_, actor_loss_curve = plt.subplots()
			actor_loss_curve.plot(a_losses)
			plt.savefig('training_plots/TD3_BC/actor_losses_training_curve.png')

			_, critic_loss_curve = plt.subplots()
			critic_loss_curve.plot(c_losses)
			plt.savefig('training_plots/TD3_BC/critic_losses_training_curve.png')
		else:
			print("load trained model")
			policy.load("models/batch_size_256_non_weighted_sample/TD3_BC_policy")
	else:
		print(model, "is not implemented")
		policy=None
	return policy
######################### Evaluation ####################################
def evaluate_agent_performance(dataset, policy):
	rewards = []
	rewards_human = []

	for i, trial in tqdm(enumerate(dataset['trial'].unique())):
		states, actions, new_states, reward_, terminals= get_trajectory(dataset, trial)
		
		state = states[0][:]
		pi_e = policy.select_action(state)
		cuevel = pi_e[1:]
		state_cueball = state[0:2]
		state_cue_front = state[8:10]
		state_cue_back = state[10:12]

		reward_human = reward_[terminals == True]
		reward = 0
		for k in range(states.size(dim=0)):
			#MSE on the action chosen by Agent on each state of a trajectory from the behaviour dataset
			next_state_cue_front, next_state_cue_back = update_cuepos(state_cue_front, state_cue_back, cuevel)
			if (next_state_cue_front[0] - state_cueball[0]).pow(2) + (next_state_cue_front[1] - state_cueball[1]).pow(2) < 0.01:	#cue tip in cueball radius
				#print("Cue stick entered radius")
				next_state_cueball = update_cueballpos(state_cueball, cuevel)
				states[k][0:2] = next_state_cueball
				states[k][2:4] = cuevel
				reward = compute_reward_evaluation(cuevel.detach().numpy(), state[6:8].detach().numpy(), reward)
			states[k][8:10] = next_state_cue_front
			states[k][10:12] = next_state_cue_back
			pi_e = policy.select_action(states[k][:])
			cuevel = pi_e[1:]
		rewards = np.append(rewards, reward)
		rewards_human = np.append(rewards_human, reward_human)
	print("number of successful trials agent: ", np.count_nonzero(rewards), rewards)
	print("number of successful trials human subject: ", np.count_nonzero(rewards_human), rewards_human)

def evaluate_close_to_human_behaviour(dataset, policy, iteration):
	#losses = []
	#avg_loss = 0.0

	## Actions Visualisation
	epoch_loss_ = []
	epoch_loss_u_ = []
	behaviour_actions_ = []
	agent_actions_ = []
	behaviour_actions_u_ = []
	agent_actions_u_ = []
	count_successful_trajectory = 0
	count_unsuccessful_trajectory = 0
	successful_trajectory = []
	unsuccessful_trajectory = []


	angle_b_ = []
	vel_x_b_ = []
	vel_z_b_ = []

	angle_e_ = []
	vel_x_e_ = []
	vel_z_e_ = []

	angle_b_u_ = []
	vel_x_b_u_ = []
	vel_z_b_u_ = []

	angle_e_u_ = []
	vel_x_e_u_ = []
	vel_z_e_u_ = []

	for i, trial in tqdm(enumerate(dataset['trial'].unique())):
		loss = 0.0
		#states, actions, new_states, reward, terminals= sample(dataset)
		#print("not terminals: ", terminals)
		states, actions, new_states, reward, terminals= get_trajectory(dataset, trial)
		if torch.any(terminals) == True:
			if reward[terminals] != -10.0:
				count_successful_trajectory += 1
				successful_trajectory = np.append(successful_trajectory, i)
				## loss
				epoch_loss = np.zeros(states.size(dim=0))
				## Actions Visualisation
				behaviour_actions = np.zeros((states.size(dim=0),3))
				agent_actions = np.zeros((states.size(dim=0),3))
				angle_b = np.zeros((states.size(dim=0)))
				vel_x_b = np.zeros((states.size(dim=0)))
				vel_z_b = np.zeros((states.size(dim=0)))

				angle_e = np.zeros((states.size(dim=0)))
				vel_x_e = np.zeros((states.size(dim=0)))
				vel_z_e = np.zeros((states.size(dim=0)))
				## Actions Visualisation

				for k in range(states.size(dim=0)):
					#MSE on the action chosen by Agent on each state of a trajectory from the behaviour dataset
					pi_e = policy.select_action(states[k][:])
					pi_b = actions[k][:]
					#print("step ", k, pi_e, pi_b)
					epoch_loss[k] = F.mse_loss(pi_e, pi_b)
					loss += F.mse_loss(pi_e, pi_b)

					## Actions Visualisation
					angle_b[k] = pi_b[0].detach().numpy()
					vel_x_b[k] = pi_b[1].detach().numpy()
					vel_z_b[k] = pi_b[2].detach().numpy()

					angle_e[k] = pi_e[0].detach().numpy()
					vel_x_e[k] = pi_e[1].detach().numpy()
					vel_z_e[k] = pi_e[2].detach().numpy()

					behaviour_actions[k,:] = pi_b.detach().numpy()#[1]
					agent_actions[k,:] = pi_e.detach().numpy()#[1]


				angle_b_ = np.append(angle_b_, angle_b)
				vel_x_b_ = np.append(vel_x_b_, vel_x_b)
				vel_z_b_ = np.append(vel_z_b_, vel_z_b)

				angle_e_ = np.append(angle_e_, angle_e)
				vel_x_e_ = np.append(vel_x_e_, vel_x_e)
				vel_z_e_ = np.append(vel_z_e_, vel_z_e)

				epoch_loss_ = np.append(epoch_loss_ , epoch_loss)
				#behaviour_actions_ = np.append(behaviour_actions_, behaviour_actions)
				#agent_actions_ = np.append(agent_actions_, agent_actions)

			
			elif reward[terminals] == -10.0:
				count_unsuccessful_trajectory += 1
				unsuccessful_trajectory = np.append(unsuccessful_trajectory, i)
				## loss
				epoch_loss_u = np.zeros(states.size(dim=0))
				## Actions Visualisation
				angle_b_u = np.zeros((states.size(dim=0)))
				vel_x_b_u = np.zeros((states.size(dim=0)))
				vel_z_b_u = np.zeros((states.size(dim=0)))

				angle_e_u = np.zeros((states.size(dim=0)))
				vel_x_e_u = np.zeros((states.size(dim=0)))
				vel_z_e_u = np.zeros((states.size(dim=0)))
				## Actions Visualisation

				for k in range(states.size(dim=0)):
					#MSE on the action chosen by Agent on each state of a trajectory from the behaviour dataset
					pi_e_u = policy.select_action(states[k][:])
					pi_b_u = actions[k][:]
					epoch_loss_u[k] = F.mse_loss(pi_e_u, pi_b_u)
					loss += F.mse_loss(pi_e_u, pi_b_u)

					## Actions Visualisation
					angle_b_u[k] = pi_b_u[0].detach().numpy()
					vel_x_b_u[k] = pi_b_u[1].detach().numpy()
					vel_z_b_u[k] = pi_b_u[2].detach().numpy()

					angle_e_u[k] = pi_e_u[0].detach().numpy()
					vel_x_e_u[k] = pi_e_u[1].detach().numpy()
					vel_z_e_u[k] = pi_e_u[2].detach().numpy()


				angle_b_u_ = np.append(angle_b_u_, angle_b_u)
				vel_x_b_u_ = np.append(vel_x_b_u_, vel_x_b_u)
				vel_z_b_u_ = np.append(vel_z_b_u_, vel_z_b_u)

				angle_e_u_ = np.append(angle_e_u_, angle_e_u)
				vel_x_e_u_ = np.append(vel_x_e_u_, vel_x_e_u)
				vel_z_e_u_ = np.append(vel_z_e_u_, vel_z_e_u)

				epoch_loss_u_ = np.append(epoch_loss_u_ , epoch_loss_u)
		else:
			print("No terminal state in trial ", trial, terminals)
		## Actions Visualisation

	#print("number of test trials: ", len(dataset['trial'].unique()), "count_unsuccessful_trajectory: ", count_unsuccessful_trajectory, "count_successful_trajectory: ", count_successful_trajectory)
	
	X_s = []
	X_u = []
	
	fig, mean_b = plt.subplots()
	mean_b.set_xlabel("timesteps")
	mean_b.set_ylabel("Angle")
	fig.suptitle('Behaviour and Agent policy', fontsize=16, y=1.04)
	#X = np.arange(1, len(dataset['trial'])+1) 
	for i in range(count_successful_trajectory):   
		X_s = np.arange(successful_trajectory[i]*140, (successful_trajectory[i]+1)*140)
		mean_b.plot(X_s, angle_b_[i*140:(i+1)*140], color='blue')
		mean_b.plot(X_s, angle_e_[i*140:(i+1)*140], color='green')
	for i in range(count_unsuccessful_trajectory):
		X_u = np.arange(unsuccessful_trajectory[i]*140, (unsuccessful_trajectory[i]+1)*140)
		mean_b.plot(X_u, angle_b_u_[i*140:(i+1)*140], color='blue')
		mean_b.plot(X_u, angle_e_u_[i*140:(i+1)*140], color='red')
	#mean_b.legend()
	#fig.tight_layout()
	fig.savefig('training_plots/TD3_BC/iter_'+str(iteration)+'_angle_behaviour_agent.png')

	fig, mean_b = plt.subplots()
	mean_b.set_xlabel("timesteps")
	mean_b.set_ylabel("Velocity x-axis")
	fig.suptitle('Behaviour and Agent policy', fontsize=16, y=1.04)
	for i in range(count_successful_trajectory):   
		X_s = np.arange(successful_trajectory[i]*140, (successful_trajectory[i]+1)*140)
		mean_b.plot(X_s, vel_x_b_[i*140:(i+1)*140], color='blue')
		mean_b.plot(X_s, vel_x_e_[i*140:(i+1)*140], color='green')
	for i in range(count_unsuccessful_trajectory):
		X_u = np.arange(unsuccessful_trajectory[i]*140, (unsuccessful_trajectory[i]+1)*140)
		mean_b.plot(X_u, vel_x_b_u_[i*140:(i+1)*140], color='blue')
		mean_b.plot(X_u, vel_x_e_u_[i*140:(i+1)*140], color='red')
	#mean_b.legend()
	#fig.tight_layout()
	fig.savefig('training_plots/TD3_BC/iter_'+str(iteration)+'_vel_x_behaviour_agent.png')

	fig, mean_b = plt.subplots()
	mean_b.set_xlabel("timesteps")
	mean_b.set_ylabel("Velocity z-axis")
	fig.suptitle('Behaviour and Agent policy', fontsize=16, y=1.04)
	for i in range(count_successful_trajectory):   
		X_s = np.arange(successful_trajectory[i]*140, (successful_trajectory[i]+1)*140)
		mean_b.plot(X_s, vel_z_b_[i*140:(i+1)*140], color='blue')
		mean_b.plot(X_s, vel_z_e_[i*140:(i+1)*140], color='green')
	for i in range(count_unsuccessful_trajectory):
		X_u = np.arange(unsuccessful_trajectory[i]*140, (unsuccessful_trajectory[i]+1)*140)
		mean_b.plot(X_u, vel_z_b_u_[i*140:(i+1)*140], color='blue')
		mean_b.plot(X_u, vel_z_e_u_[i*140:(i+1)*140], color='red')
	#mean_b.legend()
	#fig.tight_layout()
	fig.savefig('training_plots/TD3_BC/iter_'+str(iteration)+'_vel_z_behaviour_agent.png')

	fig, mean_b = plt.subplots()
	mean_b.set_xlabel("timesteps")
	mean_b.set_ylabel("MSE loss")
	fig.suptitle('MSE loss between Behaviour and Agent policy', fontsize=16, y=1.04)  
	for i in range(count_successful_trajectory):   
		X_s = np.arange(successful_trajectory[i]*140, (successful_trajectory[i]+1)*140)
		mean_b.plot(X_s, epoch_loss_[i*140:(i+1)*140], color='green')
	for i in range(count_unsuccessful_trajectory):
		X_u = np.arange(unsuccessful_trajectory[i]*140, (unsuccessful_trajectory[i]+1)*140)
		mean_b.plot(X_u, epoch_loss_u_[i*140:(i+1)*140], color='red')
	#fig.tight_layout()
	fig.savefig('training_plots/TD3_BC/iter_'+str(iteration)+'_MSE_loss_between_actions.png')

	'''## Actions Visualisation
	behaviour_actions_graph = np.zeros((count_successful_trajectory, 50))    
	agent_actions_graph = np.zeros((count_successful_trajectory, 50))   
	
	for i in range(count_successful_trajectory):
		behaviour_actions_graph[i][:] = behaviour_actions_[i*50:(i+1)*50]   
		agent_actions_graph[i][:] = agent_actions_[i*50:(i+1)*50] 
	behaviour_actions_mean = np.mean(behaviour_actions_graph, axis=0)
	behaviour_actions_std = np.std(behaviour_actions_graph, axis=0)
	agent_actions_mean = np.mean(agent_actions_graph , axis=0)
	agent_actions_std = np.std(agent_actions_graph , axis=0)

	behaviour_actions_graph_u = np.zeros((count_unsuccessful_trajectory, 50))   
	agent_actions_graph_u = np.zeros((count_unsuccessful_trajectory, 50)) 
	
	for i in range(count_unsuccessful_trajectory):
		behaviour_actions_graph_u[i][:] = behaviour_actions_u_[i*50:(i+1)*50]
		agent_actions_graph_u[i][:] = agent_actions_u_[i*50:(i+1)*50]
	behaviour_actions_mean_u = np.mean(behaviour_actions_graph_u, axis=0)
	behaviour_actions_std_u = np.std(behaviour_actions_graph_u, axis=0)
	agent_actions_mean_u = np.mean(agent_actions_graph_u , axis=0)
	agent_actions_std_u = np.std(agent_actions_graph_u , axis=0)
	print("Agent size ", agent_actions_mean_u.shape)

	fig, mean_b = plt.subplots()
	mean_b.set_xlabel("timesteps")
	mean_b.set_ylabel("Impulse force")
	fig.suptitle('Behaviour and Agent policy on Successful trial', fontsize=16, y=1.04)
	X = np.arange(1, 50+1)   
	mean_b.errorbar(X, agent_actions_mean, agent_actions_std, color='blue', ecolor = 'red')#, linestyle='None', marker='^')
	mean_b.plot(X, behaviour_actions_mean, color='green')
	fig.savefig('training_plots/Successful_behaviour_agent.png')
	
	fig1, mean_b_1 = plt.subplots()
	mean_b_1.set_xlabel("timesteps")
	mean_b_1.set_ylabel("Impulse force")
	fig1.suptitle('Behaviour and Agent policy on Unsuccessful trial', fontsize=16, y=1.04)
	X = np.arange(1, 50+1)
	mean_b_1.errorbar(X, agent_actions_mean_u, agent_actions_std_u, color='blue', ecolor = 'red')#, linestyle='None', marker='^')
	mean_b_1.plot(X, behaviour_actions_mean_u, color='green')
	fig1.savefig('training_plots/Unsuccessful_behaviour_agent.png')
	## Actions Visualisation

	ev_fig, evaluation_loss_curve = plt.subplots()
	evaluation_loss_curve.plot(losses)
	ev_fig.savefig('training_plots/evaluation_losses.png')
	avg_loss = sum(losses)/len(losses)
	print("Average Evaluation loss: ", avg_loss)'''
	plt.close()
	return loss/len(dataset["trial"])

def plot_animated_behaviour_policy(dataset):
	states,_,_,_,_ = get_trajectory(dataset, dataset["trial"].iloc[0])
	states = states.detach().numpy()

	fig, ax = plt.subplots(1,1)

	def animate(i):
		ax.clear()
		x_values = [states[i,10], states[i,8]]
		z_values = [states[i,11], states[i,9]]
		ax.plot(x_values, z_values, color="blue", label = 'cue stick')

		x_val = states[i,0]
		z_val = states[i,1]
		ax.scatter(x_val, z_val, s=60, facecolors='black', edgecolors='black', label = 'cueball')
		
		ax.scatter(states[i][4], states[i][5], s=60, facecolors='red', edgecolors='red', label = 'target ball')
		ax.scatter(states[i][6], states[i][7], s=400, facecolors='none', edgecolors='green', label = 'pocket')
		ax.set(xlim=(-2, 2), ylim=(-2, 2))
		ax.legend()


	anim = animation.FuncAnimation(fig, animate,  frames = len(states), interval=20, repeat=False)	
	plt.close()
	anim.save('training_plots/Agent_policy3.gif', writer='imagemagick')	
	print("saved GIF")

def plot_animated_learnt_policy(dataset, policy, iteration=44):

	states,_,_,_,_ = get_trajectory(dataset, dataset["trial"].iloc[31])
	states = states.detach().numpy()
	fig, ax = plt.subplots(1,1)
	
	#actions = policy.select_action(states).detach().numpy()
	
	
	'''fig, mean_b = plt.subplots()
	mean_b.set_xlabel("timesteps")
	mean_b.set_ylabel("Vel")
	fig.suptitle('Behaviour and Agent policy', fontsize=16, y=1.04)
	X = np.arange(1, len(states)+1)
	#mean_b.plot(X, actions[:,0], color='green', label='angle')
	mean_b.plot(X, actions[:,1], color='blue', label='vel_x')#, linestyle='None', marker='^')
	mean_b.plot(X, actions[:,2], color='red', label='vel_z')
	mean_b.legend()
	fig.tight_layout()
	fig.savefig('training_plots/actions.png')
	plt.close()'''

	cueposfront = np.zeros((states.shape[0],2))
	cueposback = np.zeros((states.shape[0],2))
	cueposfront[0][0] = states[0,8]
	cueposfront[0][1] = states[0,9]
	cueposback[0][0] = states[0,10]
	cueposback[0][1] = states[0,11]
	actions = policy.select_action(states[0,:]).detach().numpy()

	dt = 0.0111
	for i in range(1, actions.shape[0]-1):
		#cueposfront[i][:], cueposback[i][:] = update_cuepos(cueposfront[i-1][:], cueposback[i-1][:], cuevel[i][:])
		cueposfront[i][:] = cueposfront[i-1][:] + actions[1:]*dt
		cueposback[i][:] = cueposback[i-1][:] + actions[1:]*dt
		#update states
		states[i,8] = cueposfront[i][0]
		states[i,9] = cueposfront[i][1]
		states[i,10] = cueposback[i][0]
		states[i,11] = cueposback[i][1]
		actions = policy.select_action(states[i,:]).detach().numpy()

	cueballpos = np.zeros((states.shape[0],2))
	cueballpos[0][0] = states[0][0]
	cueballpos[0][1] = states[0][1]

	def animate(i):
		ax.clear()
		#x_values = [states[i,10], states[i,8]]
		x_values = [cueposback[i][0], cueposfront[i][0]]
		#z_values = [states[i,11], states[i,9]]
		z_values = [cueposback[i][1], cueposfront[i][1]]
		#line.set_data(x_values, z_values)
		ax.plot(x_values, z_values, color="blue", label = 'cue stick')

		x_val = states[i,0]	#cueballpos[i,0]	
		z_val = states[i,1]	#cueballpos[i,1]	
		ax.scatter(x_val, z_val, s=60, facecolors='black', edgecolors='black', label = 'cueball')
		
		ax.scatter(states[i][4], states[i][5], s=60, facecolors='red', edgecolors='red', label = 'target ball')
		ax.scatter(states[i][6], states[i][7], s=400, facecolors='none', edgecolors='green', label = 'pocket')
		ax.set(xlim=(-2, 2), ylim=(-2, 2))
		ax.legend()

		'''ax.scatter(0, 0, ms=7,facecolors='none', edgecolors='none', label = str(i))
		ax.plot(1, 1, ms=7,facecolors='none', edgecolors='none', label = str(actions[i][0]))'''


	anim = animation.FuncAnimation(fig, animate,  frames = len(states), interval=20, repeat=False)	
	plt.close()
	anim.save('training_plots/TD3_BC/iter_'+str(iteration)+'_Agent_learnt_policy.gif', writer='imagemagick')	
	#print("saved GIF")
########################## Main  ########################################

if __name__ == "__main__":

    filename = ["RL_dataset/Offline_reduced/AAB_Offline_reduced.csv"]	#"RL_dataset/Offline_reduced/AAB_Offline_reduced.csv", "RL_dataset/Offline_reduced/AS_Offline_reduced.csv","RL_dataset/Offline_reduced/BL_Offline_reduced.csv"]
    '''["RL_dataset/Offline_reduced/AAB_Offline_reduced.csv", "RL_dataset/Offline_reduced/AS_Offline_reduced.csv", "RL_dataset/Offline_reduced/BL_Offline_reduced.csv",
    "RL_dataset/Offline_reduced/BR_Offline_reduced.csv", "RL_dataset/Offline_reduced/BY_Offline_reduced.csv", "RL_dataset/Offline_reduced/CP_Offline_reduced.csv",
     "RL_dataset/Offline_reduced/DR_Offline_reduced.csv", "RL_dataset/Offline_reduced/DS_Offline_reduced.csv",
    "RL_dataset/Offline_reduced/DZ_Offline_reduced.csv", "RL_dataset/Offline_reduced/ESS_Offline_reduced.csv",
    "RL_dataset/Offline_reduced/KO_Offline_reduced.csv", "RL_dataset/Offline_reduced/LR_Offline_reduced.csv",
    "RL_dataset/Offline_reduced/JW_Offline_reduced.csv", "RL_dataset/Offline_reduced/IK_Offline_reduced.csv", "RL_dataset/Offline_reduced/HZ_Offline_reduced.csv",
    "RL_dataset/Offline_reduced/GS_Offline_reduced.csv"]'''	

    train_set, test_set = load_clean_data(filename)
	
    policy = train(train_set, test_set, state_dim=14, action_dim=3, epochs=10000, train=True, model="TD3_BC")	#TD3_BC #set train=False to load pretrained model


    #evaluate(train_set, policy)
    #plot_animated_behaviour_policy(test_set)
    #plot_animated_learnt_policy(test_set, policy)
