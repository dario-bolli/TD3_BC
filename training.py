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
					'terminals': df['terminals'][train_ind]}

	test_set = {'trial': trial[test_ind],
					'states': states[test_ind],
					'actions': actions[test_ind],
					'new_states': new_states[test_ind],
					'rewards': df['rewards'][test_ind],
					'terminals': df['terminals'][test_ind]}
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
	print("Train set number of trajectories: ", train_set['trial'].unique().shape, "Test set number of trajectories: ",test_set['trial'].unique().shape)
	return train_set, test_set
################################################### Agent ##################################################################

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 512)
		self.l3 = nn.Linear(512, 256)
		self.l4 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		a = F.relu(self.l3(a))
		return self.max_action * torch.tanh(self.l4(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 512)
		self.l3 = nn.Linear(512, 256)
		self.l4 = nn.Linear(256, 1)

		# Q2 architecture
		self.l5 = nn.Linear(state_dim + action_dim, 256)
		self.l6 = nn.Linear(256, 512)
		self.l7 = nn.Linear(512, 256)
		self.l8 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = F.relu(self.l3(q1))
		q1 = self.l4(q1)

		q2 = F.relu(self.l5(sa))
		q2 = F.relu(self.l6(q2))
		q2 = F.relu(self.l7(q2))
		q2 = self.l8(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = F.relu(self.l3(q1))
		q1 = self.l4(q1)
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
		loss = 0
		# Sample replay buffer 
		states, actions, new_states, rewards, not_done = trajectory
		"""states = trajectory[0]
		actions = trajectory[1] 
		new_states = trajectory[2] 
		rewards = trajectory[3] 
		not_done = trajectory[4]"""
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
				loss += actor_loss
				# Optimize the actor 
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()

				# Update the frozen target models
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		return loss/states.size(dim=0)

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

######################### Train ####################################
def train(dataset, state_dim=24, action_dim=2, epochs=3000):  
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

	batch_size = 256
	loss_values = []

	for i in tqdm(range(1, epochs)):
		trajectory = sample(dataset)
		loss = policy.train(trajectory, batch_size)	#, args, **kwargs
		loss_values.append(loss)
		
		if i%10 == 0:
			policy.save("models/TD3_BC_policy")
	
	#plt.plot(loss_values)
	return policy
######################### Evaluation ####################################
def evaluate(dataset, policy):
	print("start Evaluation")
	loss = 0
	avg_loss = 0
	for i, trial in tqdm(enumerate(dataset['trial'].unique())):
		states, actions, _,_,_ = get_trajectory(dataset, trial)	#new_states, rewards, not_done

		for j in range(states.size(dim=0)):

			#MSE on the action chosen by Agent on each state of a trajectory from the behaviour dataset
			pi_e = policy.select_action(states[j][:])
			pi_b = actions[j][:]
			loss += F.mse_loss(pi_e, pi_b)
			#MSE on final result of a trajectory?
			if j==states.size(dim=0)-1:
				print(pi_e, pi_b)
		loss = loss/states.size(dim=0)
		avg_loss += loss
		print("Loss for trajectory ", i, " (trial number: ", trial, ") = ", loss)
	avg_loss = avg_loss/len(dataset['trial'].unique())
	print("Average Evaluation loss: ", avg_loss)

########################## Main  ########################################

if __name__ == "__main__":
	filename = ["RL_dataset/Offline_reduced/AAB_Offline_reduced.csv",
	 "RL_dataset/Offline_reduced/AE_Offline_reduced.csv", "RL_dataset/Offline_reduced/AK_Offline_reduced.csv",
	 "RL_dataset/Offline_reduced/AS_Offline_reduced.csv"]	#"RL_dataset/Offline_reduced/AAB_Offline_reduced.csv" #"RL_dataset/AAB.csv"
	train_set, test_set = load_data(filename)
	#state_dim=24, action_dim=2  
	#state dim must be multiple of 6
	policy = train(train_set,state_dim=14, action_dim=2, epochs=30000)
	evaluate(test_set, policy)