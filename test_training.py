import os
import tqdm
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt

#import seaborn as sns
#import statsmodels.api as sm

import scipy.stats as ss
#import pingouin as pg

import scipy.signal as sci
import math
import scipy as sp

import itertools

from datetime import datetime, timedelta
from datetime import date

import re

import copy
import argparse
#import d4rl
#from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################# Functions ################################################
class Offline_RL_dataset(object): #rewards,cueballpos,redballpos, targetcornerpos, cueposfront, cueposback, cuedirection, cuevel,
    def __init__(self, state_dim=14, action_dim=2, nb_trial=250):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #number of trial
        self.N = nb_trial    #int(cueballpos[0].iloc[-1])
        self.state_dim = state_dim  
        self.action_dim = action_dim
        self.trajectories = []
        
        self.mean = np.zeros(self.state_dim)
        self.std = np.zeros(self.state_dim)

    def get_trajectories(self, list_data, rewards):
        
        state_ = []
        new_state_ = []
        action_ = []
        reward_ = []
        done_ = []

        state = np.zeros(self.state_dim)
        new_state = np.zeros(self.state_dim) 
        action = np.zeros(self.action_dim)
        reward = np.zeros(1)

        actions = self.compute_impulseForce_load(np.transpose(np.array([list_data["cuevel"][1], list_data["cuevel"][3]])), np.transpose(np.array([list_data["cuedirection"][1], list_data["cuedirection"][3]])))

        for i in range(len(list_data["cueballpos"])-1):
            count=0
            for x in list_data:
                state[count] = list_data[str(x)][1].iloc[i]
                state[count+1] = list_data[str(x)][3].iloc[i]
                new_state[count] = list_data[str(x)][1].iloc[i+1]
                new_state[count+1] = list_data[str(x)][3].iloc[i+1]
                count+=2
            #Action Velocity, Force?
            action = actions[i][:]
            reward = rewards.iloc[i]
            
            if cueballpos[0].iloc[i+1] != cueballpos[0].iloc[i]:
                done_bool = True
            else:
                done_bool = False

            state_.append(state.copy())
            new_state_.append(new_state.copy())
            action_.append(action.copy())
            reward_.append(reward.copy())
            done_.append(done_bool)

            if list_data["cueballpos"][0].iloc[i+1] != list_data["cueballpos"][0].iloc[i]:
                self.trajectories.append({
                'size': np.array(state_).shape[0],
                'states': np.array(state_),
                'actions': np.array(action_),
                'new_states': np.array(new_state_),
                'rewards': np.array(reward_),
                'terminals': np.array(done_),
                })

    def sample(self):   #, batch_size = 4):   #bacth size is the number of trajectory processed before gradient update
        max_size = self.N
        ind = 1 #np.random.randint(0, max_size)    #, size=batch_size)
        return (torch.Tensor.int(self.trajectories[ind]['size']).to(self.device),
            torch.FloatTensor(self.trajectories[ind]['states']).to(self.device),
            torch.FloatTensor(self.trajectories[ind]['actions']).to(self.device),
            torch.FloatTensor(self.trajectories[ind]['new_states']).to(self.device),
            torch.FloatTensor(self.trajectories[ind]['rewards']).to(self.device),
            torch.Tensor.bool(self.trajectories[ind]['terminals']).to(self.device))

    def compute_impulseForce_load(self, cuevel, cuedirection):
        impulseForce = np.zeros(cuevel.shape)       #(N,2)
        shotMagnitude = np.zeros(1)
        shotDir = np.zeros(cuedirection.shape)
        #Reward: magnitude range
        lbMagnitude = 0.4   #0.516149
        ubMagnitude = 0.882607

        shotMagnitude = np.linalg.norm(cuevel, axis=1)

        for i in range(cuevel.shape[0]):
            if shotMagnitude[i] > ubMagnitude:
                shotMagnitude[i] = ubMagnitude
                #print("upper bounded")
            #elif shotMagnitude[i] > lbMagnitude:
                #print(i, shotMagnitude[i])
            elif shotMagnitude[i] < lbMagnitude:
                shotMagnitude[i] = 0

            shotDir[i][:] = cuedirection[i][:]
            if shotMagnitude[i] == 0:
                impulseForce[i][:] = 0
            else:
                impulseForce[i][:] = shotMagnitude[i] * shotDir[i][:]
        return impulseForce

    def compute_mean_std(self, list_data, eps = 1e-3):
        count = 0
        for i,x in enumerate(list_data):
            self.mean[count] = list_data[str(x)][1].mean(0)  #, keepdim=True)
            self.mean[count+1] = list_data[str(x)][3].mean(0)

            self.std[count] = list_data[str(x)][1].std(0)  #, keepdim=True)
            self.std[count+1] = list_data[str(x)][3].std(0)
            count += 2

    def normalize_states(self, eps = 1e-3):
        for i in range(2):  #self.N
            for j in range(self.state_dim):
                self.trajectories[i]['states'][:][j] = (self.trajectories[i]['states'][:][j] - self.mean[j])/self.std[j]
                self.trajectories[i]['new_states'][:][j] = (self.trajectories[i]['new_states'][:][j] - self.mean[j])/self.std[j]


################################### TD3_BC AGENT ##############################################

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
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
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, trajectory, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		size, states, actions, new_states, rewards, not_done = trajectory
		print(size)
		
		for i in range(states.size(dim=0)):
			with torch.no_grad():
				# Select action according to policy and add clipped noise
				noise = (
					torch.randn_like(actions[i][:]) * self.policy_noise
				).clamp(-self.noise_clip, self.noise_clip)
				
				next_action = (
					self.actor_target(new_states[i][:]) + noise
				).clamp(-self.max_action, self.max_action)

				# Compute the target Q value
				target_Q1, target_Q2 = self.critic_target(new_states[i][:], next_action)
				target_Q = torch.min(target_Q1, target_Q2)
				target_Q = rewards[i] + not_done * self.discount * target_Q

			# Get current Q estimates
			current_Q1, current_Q2 = self.critic(states[i][:], actions[i][:])

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
				pi = self.actor(states[i][:])
				Q = self.critic.Q1(states[i][:], pi)
				lmbda = self.alpha/Q.abs().mean().detach()

				actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, actions[i][:]) 
				
				print("iteration ", self.total_it, " actor_loss: ", actor_loss)

				# Optimize the actor 
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()

				# Update the frozen target models
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


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

######### Train ############
def train(list_data, args, **kwargs):  
	# Environment State Properties
	corner = "all"
	state_dim=14
	action_dim=2
	max_action = 1
	normalize = True
    # Agent parameters
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

	train_set = Offline_RL_dataset()
	train_set.get_trajectories(list_data, rewards)
	train_set.compute_mean_std(list_data)
	train_set.normalize_states()

	batch_size = 4	#64 #256

	for i in range(1, 100):
		print("episode: ", i)
		trajectory = train_set.sample()
		policy.train(trajectory, batch_size)	#, args, **kwargs

		if i%10 == 0:
			policy.save("TD3_BC_policy")
  

########################## Main  ########################################

if __name__ == "__main__":
    # Read Saved dataset
    df = pd.read_csv("RL_dataset/AAB_raw_data.csv", header = 0, \
            names = ['rewards','cueballpos','redballpos', 'targetcornerpos', 'cueposfront', 'cueposback', 'cuedirection', 'cuevel'], usecols = [1,2,3,4,5,6,7,8], lineterminator = "\n")
    df = df.replace([r'\n', r'\[', r'\]'], '', regex=True) 
    rewards = pd.DataFrame.from_records(np.array(df['rewards'].astype(str).str.split(','))).astype(float)
    cueballpos = pd.DataFrame.from_records(np.array(df['cueballpos'].str.split(','))).astype(float)
    redballpos = pd.DataFrame.from_records(np.array(df['redballpos'].str.split(','))).astype(float)
    targetcornerpos = pd.DataFrame.from_records(np.array(df['targetcornerpos'].str.split(','))).astype(float)
    cueposfront = pd.DataFrame.from_records(np.array(df['cueposfront'].str.split(','))).astype(float)
    cueposback = pd.DataFrame.from_records(np.array(df['cueposback'].str.split(','))).astype(float)
    cuedirection = pd.DataFrame.from_records(np.array(df['cuedirection'].str.split(','))).astype(float)
    cuevel = pd.DataFrame.from_records(np.array(df['cuevel'].str.split(','))).astype(float)

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
			"state_dim": 14,
			"action_dim": 3,
			"max_action": 1,
			"discount": 0.99,
			"tau": 0.005,
			# TD3
			"policy_noise": 0.2,    #args.policy_noise * max_action,
			"noise_clip": 0.5,  #args.noise_clip * max_action,
			"policy_freq": 10,
			# TD3 + BC
			"alpha": 2.5
		}

    print("Offline RL dataset")
    list_data = {'cueballpos': cueballpos,
             'redballpos': redballpos, 
             'targetcornerpos': targetcornerpos,
             'cueposfront': cueposfront, 
             'cueposback': cueposback,
             'cuedirection': cuedirection,
              'cuevel': cuevel}
    #dataset_RL = Offline_RL_load(list_data, rewards, cuevel, cuedirection, cueballpos)
    
    #print("Replay Buffer")
    #state_dim=14
    #action_dim=2
    #replay_buffer = ReplayBuffer(state_dim, action_dim)
    #replay_buffer.convert_D4RL(dataset_RL)

    print("start training")
    train(list_data, args, **kwargs)