import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
from torch.nn.utils import clip_grad_norm_
import copy
import math
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################################### Functions ############################################################
def update_cueballpos(prev_pos, velocity):
    dt = 0.0111
    new_pos = prev_pos + velocity*dt
    return new_pos

def update_cuepos(prev_pos_front, actions): #, vel
    dt = 0.0111
    len_cue_stick = 1.21	#fixed value approximate over all subjects and trials, but not important to be precise, just for visualisation
    
    #acceleration = actions[10:]
    #acceleration = acceleration*(158.1916216216216)
    #new_vel = vel + acceleration*dt 

    new_vel = actions[1:]
    angle = actions[0]*(135 - 45)+45
    #print("action angle: ", angle)
    angle_complementary = 180-angle
    new_pos_front = prev_pos_front + new_vel*dt
    new_pos_back = [new_pos_front[0] + len_cue_stick*np.cos(np.radians(-angle_complementary)),0, new_pos_front[2] + len_cue_stick*np.sin(np.radians(-angle_complementary))]   #0 on y 
    
    new_cue_dir = (new_pos_front - new_pos_back)/np.linalg.norm((new_pos_front - new_pos_back))
    #print("new cue dir: ", new_cue_dir)
    return new_pos_front, new_pos_back, new_cue_dir			#, new_vel

def compute_reward_evaluation(cuevel, target_corner, reward):
    target_corner_angle = np.rad2deg(np.arctan2(target_corner[2], target_corner[0]))
    angle = np.rad2deg(np.arctan2(cuevel[2], cuevel[0]))
    if target_corner_angle < 90.0:
        angle = 180 - angle
    mean_pocket = 105.0736
    lbPocket = mean_pocket - 0.5	#0.3106383 #= 104.7629617
    ubPocket = mean_pocket + 0.5	#0.2543617 #= 105.3279617

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
    ind2 = np.random.choice(dic['trial'][dic['terminals'] == True],p=prob_dist)
    ind3 = np.random.choice(dic['trial'][dic['terminals'] == True],p=prob_dist)
    ind4 = np.random.choice(dic['trial'][dic['terminals'] == True],p=prob_dist)

    mask1 = dic['trial'] == ind
    mask2 = dic['trial'] == ind2
    mask3 = dic['trial'] == ind3
    mask4 = dic['trial'] == ind4

    mask = mask1+mask2+mask3+mask4
    return (torch.FloatTensor(dic['states'][mask].to_numpy()).to(device), torch.FloatTensor(dic['actions'][mask].to_numpy()).to(device), 
            torch.FloatTensor(dic['new_states'][mask].to_numpy()).to(device), torch.FloatTensor(dic['rewards'][mask].to_numpy()).to(device), 
            torch.from_numpy(dic['terminals'][mask].to_numpy(dtype=bool)).to(device))


def weighted_sample_experience(dic, sample_size):
	ind = []
	for i in range(sample_size):
		ind = np.append(ind, np.random.choice(np.arange(0,len(dic['trial']))))

	return (torch.tensor(dic['states'].iloc[ind].to_numpy(), dtype=torch.float, device=device), torch.tensor(dic['actions'].iloc[ind].to_numpy(), dtype=torch.float, device=device), 
            torch.tensor(dic['new_states'].iloc[ind].to_numpy(), dtype=torch.float, device=device), torch.tensor(dic['rewards'].iloc[ind].to_numpy(), dtype=torch.float, device=device), 
            torch.from_numpy(dic['terminals'].iloc[ind].to_numpy(dtype=bool)).to(device))

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
	trial_clean = np.concatenate((trial.unique()[0:75],trial.unique()[225:]))
	print(filename[0], "len trial: ", len(trial_clean))
	trial_ind = np.arange(0,len(trial_clean))	#select only block 1,2.3 and 10 #1,len(trial.unique()) +1 if we want 250 index as well
	train_trial_ind = np.random.choice(trial_ind, size=int(0.8*len(trial_ind)), replace=False)  #distrib proba for each value, could be useful to weight more "important" trajectories
	
	test_trial_ind = np.delete(trial_ind, train_trial_ind)
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

				trial_clean = np.concatenate((trial.unique()[0:75],trial.unique()[225:]))
				print(file, "len trial: ", len(trial_clean))
				trial_ind = np.arange(0,len(trial_clean))	#select only block 2 and 3 #1,len(trial.unique()) +1 if we want 250 index as well
				train_trial_ind = np.random.choice(trial_ind, size=int(0.8*len(trial_ind)), replace=False)  #distrib proba for each value, could be useful to weight more "important" trajectories
				
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

################################################## TD3 Agent ##################################################################

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 512)
		self.l3 = nn.Linear(512, 256)
		self.l4 = nn.Linear(256, action_dim)
		#self.actions_upper_bound = torch.tensor([1,1,1,1,1,1,1,1,0.2,1])#,1,1])    #135
		#self.actions_lower_bound = torch.tensor([0,0,0,0,0,0,-1,-1,-0.2,-1])#,-1,-1])  #45
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		a = F.relu(self.l3(a))
		a = torch.tanh(self.l4(a))
		#rescaled_actions = self.actions_lower_bound + (self.actions_upper_bound - self.actions_lower_bound) * (a + 1) / 2
		return 	a	#rescaled_actions #torch.tanh(self.l4(a))

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
		self.l6 = nn.Linear(256,512)
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
		#self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=10000, gamma=1e-5)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)	
		#self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=10000, gamma=1e-5)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.alpha = alpha

		self.total_it = 0


	def select_action(self, state):
		#state = torch.tensor(state, dtype=float, device=device)	#.reshape(1, -1)
		return self.actor(state).cpu()	#.cpu().data.numpy().flatten()


	def train(self, experience, batch_size=1):
		self.total_it += 1
		a_loss = 0.0
		c_loss = 0.0
		# Sample replay buffer 
		#states, actions, new_states, rewards, not_terminals = trajectory.sample(256)
		states, actions, new_states, rewards, terminals = experience 
		not_terminals = ~terminals#.reshape(-1,1)
		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(actions) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			next_action = (
				self.actor_target(new_states) + noise	#.reshape(-1,1)		#Warning when action shape is 1, broadcasting fucks up the addition
			).clamp(-self.max_action, self.max_action)
			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(new_states, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = rewards + not_terminals * self.discount * target_Q.squeeze()

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(states, actions)	#.reshape(-1,1))#Warning when action shape is 1
		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1.squeeze(), target_Q) + F.mse_loss(current_Q2.squeeze(), target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:   #self.total_it

			# Compute actor loss
			pi = self.actor(states)
			Q = self.critic.Q1(states, pi)
			lmbda = self.alpha/Q.abs().mean().detach()

			actor_loss = F.mse_loss(pi, actions) # -lmbda * Q.mean() +
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data) 
			a_loss += actor_loss.cpu().detach().numpy()
		c_loss += critic_loss.cpu().detach().numpy()
		#self.actor_scheduler.step()
		#self.critic_scheduler.step()
		return c_loss, a_loss

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


################################################## TD3 Off-policy ##########################################################

class AC_Off_POC_TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
			alpha=2.5,
            policy_freq=2,
            kl_div_var=0.15
        ):
        self.device = device

        # Initialize actor networks and optimizer
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=200, gamma=1e-5)

        # Initialize critic networks and optimizer
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=200, gamma=1e-5)

        # Initialize the training parameters
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

        self.action_dim = action_dim

        # Initialize the reference Gaussian
        self.kl_div_var = kl_div_var

        self.ref_gaussian = MultivariateNormal(torch.zeros(self.action_dim).to(self.device),
                                               torch.eye(self.action_dim).to(self.device) * self.kl_div_var)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)	#.reshape(1, -1)
        return self.actor(state)	#.cpu().data.numpy().flatten()

    def train(self, experience, batch_size=1):
        self.total_it += 1
        a_loss = 0.0
        c_loss = 0.0
        # Sample from the experience replay buffer
        state, action, next_state, reward, not_done = experience

        with torch.no_grad():
            # Select action according to the target policy and add target smoothing regularization
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q-value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q.squeeze()

        # Get the current Q-value estimates
        current_Q1, current_Q2 = self.critic(state, action)

        try:
            # Get current actions on the states of external experiences
            with torch.no_grad():
                current_action = self.actor(state)

            # Compute the difference batch
            diff_action_batch = action - current_action

            # Get the mean and covariance matrix for the
            mean = torch.mean(diff_action_batch, dim=0)
            cov = torch.mm(torch.transpose(diff_action_batch - mean, 0, 1), diff_action_batch - mean) / batch_size

            multivar_gaussian = MultivariateNormal(mean, cov)

            js_div = (kl_divergence(multivar_gaussian, self.ref_gaussian) + kl_divergence(self.ref_gaussian,
                                                                                          multivar_gaussian)) / 2

            js_div = torch.exp(-js_div)

            js_weights = js_div.item() * torch.ones_like(reward).to(device)
        except:
            js_weights = torch.ones_like(reward).to(device)

        # Compute the critic loss
        critic_loss = torch.sum(js_weights * (F.mse_loss(current_Q1.squeeze(), target_Q, reduction='none') +
                                              F.mse_loss(current_Q2.squeeze(), target_Q, reduction='none'))) / torch.sum(js_weights)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates, update actor networks every update period
        if self.total_it % self.policy_freq == 0:

            # Compute the actor loss
            actor_loss = -(js_weights * self.critic.Q1(state, self.actor(state))).sum() / torch.sum(js_weights)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                a_loss += actor_loss.detach().numpy()
            c_loss += critic_loss.detach().numpy()
        return c_loss, a_loss
    # Save the model parameters
    def save(self, file_name):
        torch.save(self.actor.state_dict(), file_name + "_actor")
        torch.save(self.actor_optimizer.state_dict(), file_name + "_actor_optimizer")

        torch.save(self.critic.state_dict(), file_name + "_critic")
        torch.save(self.critic_optimizer.state_dict(), file_name + "_critic_optimizer")

    # Load the model parameters
    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

################################################## CQL Agent ###############################################################

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor_CQL(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=256, hidden_size2=512, init_w=3e-3, log_std_min=-20, log_std_max=2):
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
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size)
        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
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
        
    
    def select_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        return action#.detach().cpu()
    
    def get_det_action(self, state):
        mu, log_std = self.forward(state)
        return torch.tanh(mu).detach().cpu()


class Critic_CQL(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=256, hidden_size2=512, seed=1):
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
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

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
        hidden_size2 = 512
        learning_rate = 6e-5 
        learning_rate_actor = 1e-4
        learning_rate_critic = 6e-5
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

        self.actor_local = Actor_CQL(state_size, action_size, hidden_size, hidden_size2).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate_actor)     
        
        # Critic Network (w/ Target Network)

        self.critic1 = Critic_CQL(state_size, action_size, hidden_size, hidden_size2, 2).to(device)
        self.critic2 = Critic_CQL(state_size, action_size, hidden_size, hidden_size2, 1).to(device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = Critic_CQL(state_size, action_size, hidden_size, hidden_size2).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic_CQL(state_size, action_size, hidden_size, hidden_size2).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate_critic) 

    
    def select_action(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        #state = torch.from_numpy(state).float().to(self.device)
        
        with torch.no_grad():
            if eval:
                action = self.actor_local.get_det_action(state)
            else:
                action = self.actor_local.select_action(state)
        return action.cpu()	#.detach().numpy()

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
        # Sample replay buffer 
        #states, actions, next_states, rewards, not_dones = experiences.sample(512)
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
                next_action, new_log_pi = self.actor_local.evaluate(next_states[i:i+batch_size][:])
                Q_target1_next = self.critic1_target(next_states[i:i+batch_size][:], next_action)
                Q_target2_next = self.critic2_target(next_states[i:i+batch_size][:], next_action)
                Q_target_next = torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * new_log_pi
                # Compute Q targets for current states (y_i)
                Q_targets = rewards[i:i+batch_size] + not_dones[i:i+batch_size] * self.gamma * Q_target_next.squeeze()


            # Compute critic loss
            q1 = self.critic1(states[i:i+batch_size][:], actions[i:i+batch_size][:])
            q2 = self.critic2(states[i:i+batch_size][:], actions[i:i+batch_size][:])

            critic1_loss = F.mse_loss(q1.squeeze(), Q_targets)
            critic2_loss = F.mse_loss(q2.squeeze(), Q_targets)
            
            # CQL addon
            random_actions = torch.FloatTensor(q1.shape[0] * 10, actions[i:i+batch_size][:].shape[-1]).uniform_(-1, 1).to(self.device)
            num_repeat = int (random_actions.shape[0] / states[i:i+batch_size][:].shape[0])
            temp_states = states[i:i+batch_size][:].unsqueeze(1).repeat(1, num_repeat, 1).view(states[i:i+batch_size][:].shape[0] * num_repeat, states[i:i+batch_size][:].shape[1])
            temp_next_states = next_states[i:i+batch_size][:].unsqueeze(1).repeat(1, num_repeat, 1).view(next_states[i:i+batch_size][:].shape[0] * num_repeat, next_states[i:i+batch_size][:].shape[1])
            
            current_pi_values1, current_pi_values2  = self._compute_policy_values(temp_states, temp_states)
            next_pi_values1, next_pi_values2 = self._compute_policy_values(temp_next_states, temp_states)
            
            random_values1 = self._compute_random_values(temp_states, random_actions, self.critic1).reshape(states[i:i+batch_size][:].shape[0], num_repeat, 1)
            random_values2 = self._compute_random_values(temp_states, random_actions, self.critic2).reshape(states[i:i+batch_size][:].shape[0], num_repeat, 1)
            
            current_pi_values1 = current_pi_values1.reshape(states[i:i+batch_size][:].shape[0], num_repeat, 1)
            current_pi_values2 = current_pi_values2.reshape(states[i:i+batch_size][:].shape[0], num_repeat, 1)

            next_pi_values1 = next_pi_values1.reshape(states[i:i+batch_size][:].shape[0], num_repeat, 1)
            next_pi_values2 = next_pi_values2.reshape(states[i:i+batch_size][:].shape[0], num_repeat, 1)
            
            cat_q1 = torch.cat([random_values1, current_pi_values1, next_pi_values1], 1)
            cat_q2 = torch.cat([random_values2, current_pi_values2, next_pi_values2], 1)
            
            assert cat_q1.shape == (states[i:i+batch_size][:].shape[0], 3 * num_repeat, 1), f"cat_q1 instead has shape: {cat_q1.shape}"
            assert cat_q2.shape == (states[i:i+batch_size][:].shape[0], 3 * num_repeat, 1), f"cat_q2 instead has shape: {cat_q2.shape}"
            

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

    def save(self, filename):
        torch.save(self.critic1.state_dict(), filename + "_critic1")	
        torch.save(self.critic1_optimizer.state_dict(), filename + "_critic1_optimizer")
        torch.save(self.critic2.state_dict(), filename + "_critic2")	
        torch.save(self.critic2_optimizer.state_dict(), filename + "_critic2_optimizer")

        torch.save(self.actor_local.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic1.load_state_dict(torch.load(filename + "_critic1"))
        self.critic1_optimizer.load_state_dict(torch.load(filename + "_critic1_optimizer"))
        self.critic1_target = copy.deepcopy(self.critic)
        self.critic2.load_state_dict(torch.load(filename + "_critic12"))
        self.critic2_optimizer.load_state_dict(torch.load(filename + "_critic2_optimizer"))
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor_local.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor_local)

######################### Train ####################################
def train(dataset, test_set, state_dim=14, action_dim=2, epochs=3000, train=True, model = "TD3_BC", data_folder = "data"):  
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
		print(f"Policy: {model}")
		print("---------------------------------------")
		if train == True:
			alpha_losses = []
			critic1_losses = []
			critic2_losses = []
			policy_losses = []
			evaluation_losses = []
            
			for i in tqdm(range(1, epochs)):
				experience = weighted_sample_experience(dataset, sample_size=256)   #trajectory = weighted_sample(dataset, prob_dist)    #sample(dataset) 
				
				actor_loss, alpha_loss, critic1_loss, critic2_loss, cql1_scaled_loss, cql2_scaled_loss= policy.train(experience, batch_size = 256)	#trajectory
				alpha_losses.append(alpha_loss)
				critic1_losses.append(critic1_loss) # + cql1_scaled_loss)
				critic2_losses.append(critic2_loss) # + cql2_scaled_loss)
				policy_losses.append(actor_loss)
				
				if (i%2000 == 0 and i<10000) or i%10000 == 0:
					plot_action_animated_learnt_policy(train_set, policy,"CQL_SAC",  i, data_folder)

				if (i%1000 == 0 and i<10000) or i%5000 == 0:
					#evaluate_agent_performance(test_set, policy)
					evaluation_loss = evaluate_close_to_human_behaviour(test_set, policy, "CQL_SAC",i , action_dim, data_folder)
					evaluation_losses = np.append(evaluation_losses, evaluation_loss.detach().numpy())
					if data_folder == "data":
						policy.save("models/CQL_SAC/CQL_SAC_policy")
					else:	
						policy.save("models/CQL_SAC_original/CQL_SAC_policy")
			_, alpha_loss_curve = plt.subplots()
			alpha_loss_curve.plot(alpha_losses)
			if data_folder == "data":
				plt.savefig('training_plots/CQL_SAC/alpha_losses_training_curve.png')
			else:
				plt.savefig('training_plots/CQL_SAC_original/alpha_losses_training_curve.png')

			_, critic1_loss_curve = plt.subplots()
			critic1_loss_curve.plot(critic1_losses)
			if data_folder == "data":
				plt.savefig('training_plots/CQL_SAC/critic1_losses_training_curve.png')
			else:
				plt.savefig('training_plots/CQL_SAC_original/critic1_losses_training_curve.png')

			_, critic2_loss_curve = plt.subplots()
			critic2_loss_curve.plot(alpha_losses)
			if data_folder == "data":
				plt.savefig('training_plots/CQL_SAC/critic2_losses_training_curve.png')
			else:
				plt.savefig('training_plots/CQL_SAC_original/critic2_losses_training_curve.png')

			_, policy_loss_curve = plt.subplots()
			policy_loss_curve.plot(policy_losses)
			if data_folder == "data":
				plt.savefig('training_plots/CQL_SAC/policy_losses_training_curve.png')
			else:
				plt.savefig('training_plots/CQL_SAC_original/policy_losses_training_curve.png')

			_, evaluation_loss_curve = plt.subplots()
			evaluation_loss_curve.plot(evaluation_losses)
			if data_folder == "data":
				plt.savefig('training_plots/CQL_SAC/evaluation_losses_during_training.png')
			else:
				plt.savefig('training_plots/CQL_SAC_original/evaluation_losses_during_training.png')

		else:
			print("load trained model")
			if data_folder == "data":
				policy.load("models/CQL_SAC/CQL_SAC_policy")
			else:
				policy.load("models/CQL_SAC_original/CQL_SAC_policy")

	elif model == "TD3_BC":
		# Initialize Agent
		policy = TD3_BC(**kwargs) 
		print("---------------------------------------")
		print(f"Policy: {model}")	#, Seed: {args['seed']}")
		print("---------------------------------------")
		if train == True:
			a_losses = []
			c_losses = []
			evaluation_losses = []
			

			for i in tqdm(range(1, epochs)):
				experience = weighted_sample_experience(dataset, sample_size=256)    #weighted_sample(dataset, prob_dist) #sample(dataset) 	  
				c_loss, a_loss = policy.train(experience, batch_size = 256) #replay_buffer #, args, **kwargs
				if a_loss != 0.0:	#actor loss updated only once every "policy_freq" update, set to 0 otherwise
					a_losses.append(a_loss)
				c_losses.append(c_loss)
                
				if (i%2000 == 0 and i<10000) or i%10000 == 0:
					#plot_animated_learnt_policy(test_set, policy,"TD3_BC",  i)
					#plot_3D_animated_learnt_policy(train_set, policy,"TD3_BC",  i)
					plot_action_animated_learnt_policy(train_set, policy,"TD3_BC",  i, data_folder)

				if (i%1000 == 0 and i<10000) or i%5000 == 0:
					#evaluate_agent_performance(test_set, policy)
					evaluation_loss = evaluate_close_to_human_behaviour(test_set, policy, "TD3_BC",i, action_dim, data_folder)
					evaluation_losses = np.append(evaluation_losses, evaluation_loss.detach().numpy())
					if data_folder == "data":
						policy.save("models/TD3_BC/TD3_BC_policy")
					else:
						policy.save("models/TD3_BC_original/TD3_BC_policy")
				
			_, evaluation_loss_curve = plt.subplots()
			evaluation_loss_curve.plot(evaluation_losses)
			if data_folder == "data":
				plt.savefig('training_plots/TD3_BC/evaluation_losses_during_training.png')
			else:
				plt.savefig('training_plots/TD3_BC_original/evaluation_losses_during_training.png')

			_, actor_loss_curve = plt.subplots()
			actor_loss_curve.plot(a_losses)
			if data_folder == "data":
				plt.savefig('training_plots/TD3_BC/actor_losses_training_curve.png')
			else:
				plt.savefig('training_plots/TD3_BC_original/actor_losses_training_curve.png')

			_, critic_loss_curve = plt.subplots()
			critic_loss_curve.plot(c_losses)
			if data_folder == "data":
				plt.savefig('training_plots/TD3_BC/critic_losses_training_curve.png')
			else:
				plt.savefig('training_plots/TD3_BC_original/critic_losses_training_curve.png')

		else:
			print("load trained model")
			if data_folder == "data":
				policy.load("models/TD3_BC/TD3_BC_policy")
			else:
				policy.load("models/TD3_BC_original/TD3_BC_policy")

	elif model == "AC_Off_POC_TD3":
		# Initialize Agent
		policy = AC_Off_POC_TD3(**kwargs) 
		print("---------------------------------------")
		print(f"Policy: {model}")	#, Seed: {args['seed']}")
		print("---------------------------------------")
		if train == True:
			a_losses = []
			c_losses = []
			evaluation_losses = []

			for i in tqdm(range(1, epochs)):
				policy.actor_scheduler.step()
				policy.critic_scheduler.step()
				experience = weighted_sample_experience(dataset, sample_size=256)      
				c_loss, a_loss = policy.train(experience) 
				if a_loss != 0.0:	
					a_losses.append(a_loss)
				c_losses.append(c_loss)
                
				if i%2000 == 0:
					plot_action_animated_learnt_policy(train_set, policy,"AC_Off_POC_TD3",  i)

				if i%50 == 0:
					evaluation_loss = evaluate_close_to_human_behaviour(test_set, policy, "AC_Off_POC_TD3",i, action_dim = action_dim)
					evaluation_losses = np.append(evaluation_losses, evaluation_loss.detach().numpy())
					policy.save("models/AC_Off_POC_TD3/AC_Off_POC_TD3_policy")
			
			_, evaluation_loss_curve = plt.subplots()
			evaluation_loss_curve.plot(evaluation_losses)
			plt.savefig('training_plots/AC_Off_POC_TD3/evaluation_losses_during_training.png')

			_, actor_loss_curve = plt.subplots()
			actor_loss_curve.plot(a_losses)
			plt.savefig('training_plots/AC_Off_POC_TD3/actor_losses_training_curve.png')

			_, critic_loss_curve = plt.subplots()
			critic_loss_curve.plot(c_losses)
			plt.savefig('training_plots/AC_Off_POC_TD3/critic_losses_training_curve.png')
		else:
			print("load trained model")
			policy.load("models/AC_Off_POC_TD3/TD3_BC_policy")
	else:
		print(model, "is not implemented")
		policy=None
	return policy
######################### Evaluation ####################################
def evaluate_agent_performance(dataset, policy, model="TD3_BC"):
	rewards = []
	rewards_human = []
	radius = 0.005
	vertical_margin = 0.01

	for i, trial in tqdm(enumerate(dataset['trial'].unique())):
		states, actions, new_states, reward_, terminals= get_trajectory(dataset, trial)
		
		state = states[0][:]
		
		if model == "CQL_SAC":
			actions = policy.select_action(states[0,:], eval=True)
		else:
			actions = policy.select_action(states[0,:])
		cuevel = actions[1:]
		state = states[0,:].detach().numpy()
		state_cueball = state[0:3]
		state_cue_front = state[12:15]
		state_cue_back = state[15:18]
		reward_human = reward_[terminals == True]
		reward = 0.0
		for k in range(states.shape[0]):
			#MSE on the action chosen by Agent on each state of a trajectory from the behaviour dataset
			next_state_cue_front, next_state_cue_back, next_state_cue_direction = update_cuepos(state_cue_front, actions.detach().numpy())
			if (next_state_cue_front[0] - state_cueball[0])**2 + (next_state_cue_front[2] - state_cueball[2])**2 < radius \
			and next_state_cue_front[1] < state_cueball[1] + vertical_margin:
				
				next_state_cueball = update_cueballpos(state_cueball, cuevel)
				states[k][0:3] = next_state_cueball
				reward = compute_reward_evaluation(cuevel, state[9:12], reward)
			states[k][21:] = cuevel
			states[k][12:15] = next_state_cue_front
			states[k][15:18] = next_state_cue_back
			states[k][18:21] = next_state_cue_direction
			states[k][21:] = actions[1:]
			if model == "CQL_SAC":
				actions = policy.select_action(states[k,:], eval=True)
			else:
				actions = policy.select_action(states[k,:])
			cuevel = actions[1:].detach().numpy()
		rewards = np.append(rewards, reward)
		rewards_human = np.append(rewards_human, reward_human)
	print("number of successful trials agent: ", np.count_nonzero(rewards), rewards)
	print("number of successful trials human subject: ", np.count_nonzero(rewards_human+10), rewards_human)

			
def evaluate_close_to_human_behaviour(dataset, policy, model="TD3_BC", iteration=44, action_dim=3, data_folder="data"):
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
		loss = torch.zeros([1,], dtype=torch.float64)
		states, actions, new_states, reward, terminals= get_trajectory(dataset, trial)
		if len(states) == 140:
			if torch.any(terminals) == True:
				if reward[terminals] != -10.0:
					count_successful_trajectory += 1
					successful_trajectory = np.append(successful_trajectory, i)
					## loss
					epoch_loss = np.zeros(states.size(dim=0))
					## Actions Visualisation
					behaviour_actions = np.zeros((states.size(dim=0),action_dim))
					agent_actions = np.zeros((states.size(dim=0),action_dim))
					angle_b = np.zeros((states.size(dim=0)))
					vel_x_b = np.zeros((states.size(dim=0)))
					vel_z_b = np.zeros((states.size(dim=0)))

					angle_e = np.zeros((states.size(dim=0)))
					vel_x_e = np.zeros((states.size(dim=0)))
					vel_z_e = np.zeros((states.size(dim=0)))
					## Actions Visualisation

					for k in range(states.size(dim=0)):
						#MSE on the action chosen by Agent on each state of a trajectory from the behaviour dataset
						if model == "CQL_SAC":
							pi_e = policy.select_action(states[k][:], eval=True)
						else:
							pi_e = policy.select_action(states[k][:])
						pi_b = actions[k][:].cpu()
						#print("step ", k, pi_e, pi_b)
						epoch_loss[k] = F.mse_loss(pi_e, pi_b)
						loss += F.mse_loss(pi_e, pi_b)

						## Actions Visualisation
						angle_b[k] = pi_b[0].detach().numpy()
						vel_x_b[k] = pi_b[1].detach().numpy()
						vel_z_b[k] = pi_b[2].detach().numpy()#2

						angle_e[k] = pi_e[0].detach().numpy()
						vel_x_e[k] = pi_e[1].detach().numpy()
						vel_z_e[k] = pi_e[2].detach().numpy()#2

						behaviour_actions[k,:] = pi_b.detach().numpy()#[1]
						agent_actions[k,:] = pi_e.detach().numpy()#[1]


					angle_b_ = np.append(angle_b_, angle_b)
					vel_x_b_ = np.append(vel_x_b_, vel_x_b)
					vel_z_b_ = np.append(vel_z_b_, vel_z_b)

					angle_e_ = np.append(angle_e_, angle_e)
					vel_x_e_ = np.append(vel_x_e_, vel_x_e)
					vel_z_e_ = np.append(vel_z_e_, vel_z_e)

					epoch_loss_ = np.append(epoch_loss_ , epoch_loss)

				
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
						if model == "CQL_SAC":
							pi_e_u = policy.select_action(states[k][:], eval=True)
						else:
							pi_e_u = policy.select_action(states[k][:])
						pi_b_u = actions[k][:].cpu()
						epoch_loss_u[k] = F.mse_loss(pi_e_u, pi_b_u)
						loss += F.mse_loss(pi_e_u, pi_b_u)

						## Actions Visualisation
						angle_b_u[k] = pi_b_u[0].detach().numpy()
						vel_x_b_u[k] = pi_b_u[1].detach().numpy()
						vel_z_b_u[k] = pi_b_u[2].detach().numpy()#2

						angle_e_u[k] = pi_e_u[0].detach().numpy()
						vel_x_e_u[k] = pi_e_u[1].detach().numpy()
						vel_z_e_u[k] = pi_e_u[2].detach().numpy()#2


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

	X_s = []
	X_u = []
	
	fig, mean_b = plt.subplots()
	mean_b.set_xlabel("timesteps")
	mean_b.set_ylabel("Angle")
	fig.suptitle('Behaviour and Agent policy', fontsize=16, y=1.04)
	
	for i in range(count_successful_trajectory):   
		X_s = np.arange(successful_trajectory[i]*140, (successful_trajectory[i]+1)*140)
		mean_b.plot(X_s, angle_b_[i*140:(i+1)*140], color='blue')
		mean_b.plot(X_s, angle_e_[i*140:(i+1)*140], color='green')
	for i in range(count_unsuccessful_trajectory):
		X_u = np.arange(unsuccessful_trajectory[i]*140, (unsuccessful_trajectory[i]+1)*140)
		mean_b.plot(X_u, angle_b_u_[i*140:(i+1)*140], color='blue')
		mean_b.plot(X_u, angle_e_u_[i*140:(i+1)*140], color='red')
	if data_folder == "data":
		fig.savefig('training_plots/'+model+'/iter_'+str(iteration)+'_angle_behaviour_agent.png')
	else:
		fig.savefig('training_plots/'+model+'_original/iter_'+str(iteration)+'_angle_behaviour_agent.png')

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
	if data_folder == "data":
		fig.savefig('training_plots/'+model+'/iter_'+str(iteration)+'_vel_x_behaviour_agent.png')
	else:
		fig.savefig('training_plots/'+model+'_original/iter_'+str(iteration)+'_vel_x_behaviour_agent.png')

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
	if data_folder == "data":
		fig.savefig('training_plots/'+model+'/iter_'+str(iteration)+'_vel_z_behaviour_agent.png')
	else:
		fig.savefig('training_plots/'+model+'_original/iter_'+str(iteration)+'_vel_z_behaviour_agent.png')

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
	if data_folder == "data":
		fig.savefig('training_plots/'+model+'/iter_'+str(iteration)+'_MSE_loss_between_actions.png')
	else:
		fig.savefig('training_plots/'+model+'_original/iter_'+str(iteration)+'_MSE_loss_between_actions.png')

	plt.close('all')
	return loss/len(dataset["trial"])


def plot_action_animated_learnt_policy(dataset, policy, model="TD3_BC", iteration=44, data_folder="data"):

	states,_,_,_,_ = get_trajectory(dataset, dataset["trial"].iloc[iteration])
	fig, ax = plt.subplots(1,1)
	ref_cueposfront = states[:,12:15].cpu().clone().detach()
	ref_cueposback = states[:,15:18].cpu().clone().detach()
	cueposfront = np.zeros((states.shape[0],3))
	cueposback = np.zeros((states.shape[0],3))
	cuedirection = np.zeros((states.shape[0],3))
	cueposfront[0][0] = states[0,12]
	cueposfront[0][1] = states[0,13]
	cueposfront[0][2] = states[0,14]
	cueposback[0][0] = states[0,15]
	cueposback[0][1] = states[0,16]
	cueposback[0][2] = states[0,17]
	cuedirection[0][0] = states[0,18] 
	cuedirection[0][1] = states[0,19]
	cuedirection[0][2] = states[0,20]

	if model == "CQL_SAC":
		actions = policy.select_action(states[0,:], eval=True)
	else:
		actions = policy.select_action(states[0,:])
	
	#vel = [0,0,0]
	for i in range(1, states.shape[0]-1):
		cueposfront[i][:] , cueposback[i][:], cuedirection[i][:] = update_cuepos(cueposfront[i-1][:], actions.detach().numpy())
		#cueposfront[i][:] , cueposback[i][:], vel = update_cuepos(cueposfront[i-1][:],vel, actions)
		#cueposfront2[i][:] = actions[:3]
		#cueposback2[i][:] = actions[3:6]
		#update states
		states[i,12] = cueposfront[i][0]
		states[i,13] = cueposfront[i][1]
		states[i,14] = cueposfront[i][2]
		states[i,15] = cueposback[i][0]
		states[i,16] = cueposback[i][1]
		states[i,17] = cueposback[i][2]
		states[i,18] = cuedirection[i][0]
		states[i,19] = cuedirection[i][1]
		states[i,20] = cuedirection[i][2]
		states[i,21] = actions[1]
		states[i,22] = actions[2]
		states[i,23] = actions[3]
		if model == "CQL_SAC":
			actions = policy.select_action(states[i,:], eval=True)
		else:
			actions = policy.select_action(states[i,:])

	#actions = policy.select_action(states).detach().numpy()
	#cueposfront = actions[:,1:3]
	#cueposback = actions[:,3:5]
	states = states.cpu().detach().numpy()
	cueballpos = np.zeros((states.shape[0],2))
	cueballpos[0][0] = states[0][0]
	cueballpos[0][1] = states[0][2]
	
	def animate(i):
		ax.clear()
		x_ref = [ref_cueposback[i,0], ref_cueposfront[i,0]]
		z_ref = [ref_cueposback[i,2], ref_cueposfront[i,2]]
		ax.plot(x_ref, z_ref, color="green", label = 'reference stick')


		x_values = [cueposback[i][0], cueposfront[i][0]]
		z_values = [cueposback[i][2], cueposfront[i][2]]
		ax.plot(x_values, z_values, color="blue", label = 'cue stick acc')

		
		"""x_values = [cueposback2[i][0], cueposfront2[i][0]]
		z_values = [cueposback2[i][2], cueposfront2[i][2]]
		ax.plot(x_values, z_values, color="orange", label = 'cue stick pos')"""

		x_val = states[i,0]	
		z_val = states[i,2]
		ax.scatter(x_val, z_val, s=60, facecolors='black', edgecolors='black', label = 'cueball')
		
		ax.scatter(states[i][6], states[i][8], s=60, facecolors='red', edgecolors='red', label = 'target ball')
		ax.scatter(states[i][9], states[i][11], s=200, facecolors='none', edgecolors='green', label = 'pocket')
		ax.set(xlim=(-1, 1), ylim=(-2, 2))
		ax.legend()


	anim = animation.FuncAnimation(fig, animate,  frames = len(states), interval=40, repeat=False)	
	plt.close()
	if data_folder == "data":
		anim.save('training_plots/'+model+'/iter_'+str(iteration)+'_Agent_learnt_policy.gif', writer='imagemagick')	
	else:
		anim.save('training_plots/'+model+'_original/iter_'+str(iteration)+'_Agent_learnt_policy.gif', writer='imagemagick')


########################## Main  ########################################

if __name__ == "__main__":

	filename = ["data/AAB_Offline_reduced.csv", "data/AE_Offline_reduced.csv", "data/AK_Offline_reduced.csv",
            "data/AS_Offline_reduced.csv", "data/BL_Offline_reduced.csv", "data/BR_Offline_reduced.csv",
            "data/BY_Offline_reduced.csv", "data/CP_Offline_reduced.csv", "data/CX_Offline_reduced.csv",
            "data/DR_Offline_reduced.csv", "data/DS_Offline_reduced.csv", "data/DZ_Offline_reduced.csv",
            "data/ES_Offline_reduced.csv", "data/ESS_Offline_reduced.csv", "data/GS_Offline_reduced.csv",
            "data/HL_Offline_reduced.csv", "data/HZ_Offline_reduced.csv", "data/IK_Offline_reduced.csv",
            "data/JG_Offline_reduced.csv", "data/JH_Offline_reduced.csv", "data/JR_Offline_reduced.csv",
            "data/JW_Offline_reduced.csv", "data/JY_Offline_reduced.csv", "data/KO_Offline_reduced.csv",
            "data/LR_Offline_reduced.csv", "data/MA_Offline_reduced.csv", "data/MAI_Offline_reduced.csv",
			"data/MG_Offline_reduced.csv", "data/MO_Offline_reduced.csv", "data/PM_Offline_reduced.csv",
            "data/RA_Offline_reduced.csv", "data/SC_Offline_reduced.csv", "data/SF_Offline_reduced.csv",
			"data/TA_Offline_reduced.csv", "data/TH_Offline_reduced.csv", "data/TS_Offline_reduced.csv",
            "data/VB_Offline_reduced.csv", "data/WZ_Offline_reduced.csv", "data/YC_Offline_reduced.csv",
            "data/YH_Offline_reduced.csv"]
	#["RL_dataset/Offline_reduced/AAB_Offline_reduced.csv"] 
	'''["data_original/AAB_Offline_reduced.csv", "data_original/AE_Offline_reduced.csv", "data_original/AK_Offline_reduced.csv",
            "data_original/AS_Offline_reduced.csv", "data_original/BL_Offline_reduced.csv", "data_original/BR_Offline_reduced.csv",
            "data_original/BY_Offline_reduced.csv", "data_original/CP_Offline_reduced.csv", "data_original/CX_Offline_reduced.csv",
            "data_original/DR_Offline_reduced.csv", "data_original/DS_Offline_reduced.csv", "data_original/DZ_Offline_reduced.csv",
            "data_original/ES_Offline_reduced.csv", "data_original/ESS_Offline_reduced.csv", "data_original/GS_Offline_reduced.csv",
            "data_original/HL_Offline_reduced.csv", "data_original/HZ_Offline_reduced.csv", "data_original/IK_Offline_reduced.csv",
            "data_original/JG_Offline_reduced.csv", "data_original/JH_Offline_reduced.csv", "data_original/JR_Offline_reduced.csv",
            "data_original/JW_Offline_reduced.csv", "data_original/JY_Offline_reduced.csv", "data_original/KO_Offline_reduced.csv",
            "data_original/LR_Offline_reduced.csv", "data_original/MA_Offline_reduced.csv", "data_original/MAI_Offline_reduced.csv",
			"data_original/MG_Offline_reduced.csv", "data_original/MO_Offline_reduced.csv", "data_original/PM_Offline_reduced.csv",
            "data_original/RA_Offline_reduced.csv", "data_original/SC_Offline_reduced.csv", "data_original/SF_Offline_reduced.csv",
			"data_original/TA_Offline_reduced.csv", "data_original/TH_Offline_reduced.csv", "data_original/TS_Offline_reduced.csv",
            "data_original/VB_Offline_reduced.csv", "data_original/WZ_Offline_reduced.csv", "data_original/YC_Offline_reduced.csv",
            "data_original/YH_Offline_reduced.csv"]'''

	train_set, test_set = load_clean_data(filename)
	policy = train(train_set, test_set, state_dim=24, action_dim=4, epochs=100000, train=True, model="CQL_SAC", data_folder="data")	#state_dim=16 #TD3_BC #set train=False to load pretrained model

