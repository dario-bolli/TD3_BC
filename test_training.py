import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################################### Functions ############################################################
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

				trial_clean = np.concatenate((trial.unique()[0:75],trial.unique()[224:]))
				trial_ind = np.arange(0,100)	#select only block 2 and 3 #1,len(trial.unique()) +1 if we want 250 index as well
				train_trial_ind = np.random.choice(trial_ind, size=int(0.8*len(trial_ind)), replace=False)  #distrib proba for each value, could be useful to weight more "important" trajectories
				test_trial_ind = np.delete(trial_ind, train_trial_ind)
				train_trial = trial_clean[train_trial_ind]
				test_trial = trial_clean[test_trial_ind]

				train_ind = trial.isin(train_trial)
				test_ind = trial.isin(test_trial)

				train_set['trial'] = pd.concat([train_set['trial'], trial[train_ind]], axis=0)
				train_set['states'] = pd.concat([train_set['states'], states[train_ind]], axis=0)
				train_set['actions'] = pd.concat([train_set['actions'], actions[train_ind]], axis=0)
				train_set['new_states'] = pd.concat([train_set['new_states'], new_states[train_ind]], axis=0)
				train_set['rewards'] = pd.concat([train_set['rewards'], df['rewards'][train_ind]], axis=0)
				train_set['terminals'] = pd.concat([train_set['terminals'], terminals[train_ind]], axis=0)	

				test_set['trial'] = pd.concat([test_set['trial'], trial[test_ind]], axis=0)
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
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		#a = F.relu(self.l3(a))
		return 	self.max_action * torch.tanh(self.l3(a))#torch.tanh(self.l4(a))

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
				#print("return Q estimate: ",i, target_Q)
				target_Q = rewards[i] + (not terminals[i]) * self.discount * target_Q



			# Get current Q estimates
			#print(states[i:i+batch_size][:].shape, actions[i:i+batch_size][:].shape)
			current_Q1, current_Q2 = self.critic(states[i:i+batch_size][:], actions[i:i+batch_size][:])	#.reshape(-1,1))#Warning when action shape is 1
			
			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

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

				actor_loss = -lmbda * Q.mean() #+ F.mse_loss(pi, actions[i:i+batch_size][:])
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


######################### Train ####################################
def train(dataset, state_dim=24, action_dim=2, epochs=3000, train=True, model = "TD3_BC"):  
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
	# Initialize Agent
	policy = TD3_BC(**kwargs) 
	print("---------------------------------------")
	print(f"Policy: {args['policy']}")	#, Seed: {args['seed']}")
	print("---------------------------------------")
	if train == True:
		batch_size = 256
		a_losses = []
		c_losses = []

		
		trajectory_rewards = dataset['rewards'][dataset['terminals']==True]
		prob_dist = np.ones(len(trajectory_rewards))	#len(dataset['trial'].unique())-10)#-1 Because last trial has no terminal state (to be changed in future)

		prob_dist[trajectory_rewards != -10.0] = 500	#50 times higher probs than other traj to be sampled
		prob_dist = prob_dist/prob_dist.sum()
		
		for i in tqdm(range(1, epochs)):
			trajectory = weighted_sample(dataset, prob_dist)    #sample(dataset)    

			#trajectory = sample(dataset)
			c_loss, a_loss = policy.train(trajectory, batch_size = 32)	#, args, **kwargs
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

	## Actions Visualisation
	behaviour_actions_ = []
	agent_actions_ = []
	behaviour_actions_u_ = []
	agent_actions_u_ = []
	count_successful_trajectory = 0
	count_unsuccessful_trajectory = 0
	print("len dataset: ", len(dataset['trial'].unique()))

	for i, trial in tqdm(enumerate(dataset['trial'].unique())):
		epoch_loss = 0.0
		#states, actions, new_states, reward, terminals= sample(dataset)
		#print("not terminals: ", terminals)
		states, actions, new_states, reward, terminals= get_trajectory(dataset, trial)
		if torch.any(terminals) == True:
			if reward[terminals] != -10.0:
				## Actions Visualisation
				behaviour_actions = np.zeros(states.size(dim=0))
				agent_actions = np.zeros(states.size(dim=0))
				## Actions Visualisation

				for k in range(states.size(dim=0)):
					#MSE on the action chosen by Agent on each state of a trajectory from the behaviour dataset
					pi_e = policy.select_action(states[k][:])
					pi_b = actions[k][:]
					#print("step ", k, pi_e, pi_b)
					epoch_loss += F.mse_loss(pi_e, pi_b)

					## Actions Visualisation
					behaviour_actions[k] = pi_b.detach().numpy()[1]
					agent_actions[k] = pi_e.detach().numpy()[1]
				behaviour_actions_ = np.append(behaviour_actions_, behaviour_actions)
				agent_actions_ = np.append(agent_actions_, agent_actions)
				count_successful_trajectory += 1
				## Actions Visualisation


				epoch_loss = epoch_loss/states.size(dim=0)
				losses.append(epoch_loss.detach().numpy())

			## Actions Visualisation
			elif reward[terminals] == -10.0:
				#print("terminal rewrds: ", j*trial, reward[terminals])
				behaviour_actions_u = np.zeros(states.size(dim=0))
				agent_actions_u = np.zeros(states.size(dim=0))
				for k in range(states.size(dim=0)):

					#MSE on the action chosen by Agent on each state of a trajectory from the behaviour dataset
					pi_e = policy.select_action(states[k][:])
					pi_b = actions[k][:]

					behaviour_actions_u[k] = pi_b.detach().numpy()[1]
					agent_actions_u[k] = pi_e.detach().numpy()[1]
				behaviour_actions_u_ = np.append(behaviour_actions_u_, behaviour_actions_u)
				agent_actions_u_ = np.append(agent_actions_u_, agent_actions_u)
				count_unsuccessful_trajectory += 1
		else:
			print("No terminal state in trial ", trial, terminals)
		## Actions Visualisation

	print("count_unsuccessful_trajectory: ", count_unsuccessful_trajectory, "count_successful_trajectory: ", count_successful_trajectory)
	## Actions Visualisation
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
	print("Average Evaluation loss: ", avg_loss)

########################## Main  ########################################

if __name__ == "__main__":
    filename = ["RL_dataset/AAB.csv"]
    train_set, test_set = load_clean_data(filename)
    policy = train(train_set,state_dim=14, action_dim=5, epochs=5000, train=True, model="TD3_BC")	
    evaluate(train_set, policy)
