import os
import numpy as np
import pandas as pd

########################################### Functions ######################################################################
def compute_actions(cuevel, cueDirection, cueposfront, cueposback):
    cueAngle = compute_angle(cueDirection)  #min_max_standardisation(, min_val=45, max_val=135)
    
    #cueacceleration =  normalize(compute_acceleration(cuevel[["x","y","z"]].to_numpy()))      # min_max_standardisation() #, min_val=-140, max_val=140)
    shotMagnitude = compute_impulseForce(cuevel, cueDirection)  #min_max_standardisation_acc()
    actions = np.hstack((cueAngle.reshape(-1,1) , shotMagnitude.reshape(-1,1) ))  #min_max_standardisation_acc(cuevel[["x","y","z"]].to_numpy())))   #, cueacceleration)) #, normalize(cueacceleration))  
    #vel_x = min_max_standardisation(cuevel[["x"]].to_numpy())
    #vel_y = min_max_standardisation(cuevel[["y"]].to_numpy())
    #vel_z = min_max_standardisation(cuevel[["z"]].to_numpy())
    #actions = np.hstack((cueAngle.reshape(-1,1), cueposfront[["x", "y", "z"]].to_numpy(), cueposback[["x", "y", "z"]].to_numpy(), vel_x, vel_y, vel_z))  #, cuevel.to_numpy())
    return actions  #cueAngle, cueMagnitude


def compute_cuevel(cueposfront):
    dt = 0.0111
    vel = np.zeros(cueposfront.shape)
    
    for i in range(cueposfront.shape[0]):
        if i == 0:
            vel[i][:] = 0  
        else:
            vel[i][:] = (cueposfront.iloc[i]-cueposfront.iloc[i-1])/dt
    
    return vel

def compute_acceleration(cuevel):
    dt = 0.0111
    acc = np.zeros(cuevel.shape)
    for i in range(len(cuevel)):
        if i == 0:
            acc[i][:] = 0   
        else:
            acc[i][:] = (cuevel[i][:]-cuevel[i-1][:])/dt
    return acc
        
def normalize(array):
    #array must be 1-dimensional
    min_val = array.min()
    max_val = np.abs(array).max()
    print("min and max value of acceleration standardization: ", min_val, max_val)
    array = array/max_val
    return array

def min_max_standardisation_acc(array): #=45, max_val=135 ):
    min_val = array.min()
    max_val = array.max()
    print("min and max value of actions standardization: ", min_val, max_val)
    #print("min, max: ", min_val, max_val)
    array = (array - min_val) / (max_val - min_val)
    return array

def min_max_standardisation(array, min_val=0, max_val = 0): #=45, max_val=135 ):
    #array must be 1-dimensional
    if min_val == 0:
        min_val = array.min()
        max_val = array.max()
    print("min and max value of actions standardization: ", min_val, max_val)
    #print("min, max: ", min_val, max_val)
    array = (array - min_val) / (max_val - min_val)
    return array

def compute_angle(cueDirection):
    #cueAngle1 = np.rad2deg(np.arctan2(cuevel['z'].values, cuevel['x'].values))
    #print("cue Angle vel: ", cueAngle1)
    cueAngle = np.rad2deg(np.arctan2(cueDirection["z"].values, cueDirection["x"].values))  
    #print("cue Angle Direction: ", cueAngle)
    return cueAngle

def compute_angle_for_funnel(cueDirection):
    cueAngle = np.rad2deg(np.arctan2(cueDirection.iloc[1], cueDirection.iloc[0]))      
    return cueAngle

def compute_impulseForce(cuevel, cuedirection):

    #impulseForce = np.zeros(cuevel.shape)       #(N,2)
    #shotMagnitude = np.zeros(1)
    #shotDir = np.zeros(cuedirection.shape)
    #Reward: magnitude range
    #lbMagnitude = 0.1   #0.516149
    #ubMagnitude = 3 #0.882607
    shotMagnitude = np.linalg.norm(cuevel[["x","y","z"]], axis=1)  #np.sqrt(np.square(cuevel).sum(axis=1))
    #np.linalg.norm(cuevel, axis=1)
    """for i in range(cuevel.shape[0]):
        if shotMagnitude[i] > ubMagnitude:
            shotMagnitude[i] = ubMagnitude
        elif shotMagnitude[i] < lbMagnitude:
            shotMagnitude[i] = 0

        shotDir[i][0] = cuedirection["x"].iloc[i]
        shotDir[i][1] = cuedirection["z"].iloc[i]
        if shotMagnitude[i] == 0:
            impulseForce[i][:] = 0
        else:
            impulseForce[i][:] = shotMagnitude[i] * shotDir[i][:]"""
    return shotMagnitude    #, impulseForce

#################################################################################################################

def Offline_reduced(data, cue_data, cuevel, rewards, state_dim=14, action_dim=2):   #state_dim should be a multiple of 6
    
    print("Offline RL Dataset function")
    #number of trial
    N = len(data["cueballpos"]["trial"])-1   #len(data[initial]["start_ind"])
    print(N)
    state_ = []
    new_state_ = []
    action_ = []
    reward_ = []
    done_ = []
    trial_ = []

    state = np.zeros(state_dim)  #np.zeros(21)
    new_state = np.zeros(state_dim)  #np.zeros(21)
    #action = np.zeros(13)   #12)
    #action = np.zeros(action_dim)
    #reward = np.zeros(1)
    #ball_states=6
    #cue_states=6
    #cue_states_iterations = int((state_dim-ball_states)/(cue_states))

    #Action Velocity, Force?
    #See car 2D -> action space acceleration and cueDirection
    #actions = compute_actions(cuevel[["x", "z"]], cue_data["cuedirection"][["x", "z"]], cue_data["cueposfront"][["x", "z"]], cue_data["cueposback"][["x", "z"]])
    actions = compute_actions(cuevel, cue_data["cuedirection"], cue_data["cueposfront"], cue_data["cueposback"])
   
    for i in range(N):
        count=0
        #count_action=0
        for x in data:
            state[count] = data[str(x)]["x"].iloc[i]
            state[count+1] = data[str(x)]["y"].iloc[i]
            state[count+2] = data[str(x)]["z"].iloc[i]
            new_state[count] = data[str(x)]["x"].iloc[i+1]
            new_state[count+1] = data[str(x)]["y"].iloc[i+1]
            new_state[count+2] = data[str(x)]["z"].iloc[i+1]
            count+=3

        #Add last j timesteps of the cuepos to states
        #3 observations on x and z axis makes 6 observations
        #for j in range(cue_states_iterations): #Warning if state_dim not multiple of 6, division fail for loop
        for x in cue_data:
            state[count] = cue_data[str(x)]["x"].iloc[i]
            state[count+1] = cue_data[str(x)]["y"].iloc[i]
            state[count+2] = cue_data[str(x)]["z"].iloc[i]
            new_state[count] = cue_data[str(x)]["x"].iloc[i+1]
            new_state[count+1] = cue_data[str(x)]["y"].iloc[i+1]
            new_state[count+2] = cue_data[str(x)]["z"].iloc[i+1]
            '''if x == "cueposfront" or x == "cueposback" or x == "cuevel":
                action[count_action] = cue_data[str(x)]["x"].iloc[i+1]
                action[count_action+1] = cue_data[str(x)]["y"].iloc[i+1]
                action[count_action+2] = cue_data[str(x)]["z"].iloc[i+1]
                count_action+=3'''
            count+=3
            
        #Action Velocity, Force?
        action = actions[i+1][:] #- actions[i][:]  #delta angle and delta force
        reward = rewards.iloc[i]
        
        if i == N-1:        #No last timestep due to new_states, so advance reward for the last trial
            done_bool = 1 
            reward = rewards.iloc[i+1]  
        elif data['cueballpos']['trial'].iloc[i+1] != data['cueballpos']['trial'].iloc[i]:
            done_bool = 1   #True
        else:
            done_bool = 0   #False

        trial_.append(data["cueballpos"]["trial"].iloc[i])
        state_.append(state.copy())
        new_state_.append(new_state.copy())
        action_.append(action.copy())
        reward_.append(reward)
        done_.append(done_bool)
    
    return {
        'trial': np.array(trial_),
        'states': np.array(state_),
        'actions': np.array(action_),
        'new_states': np.array(new_state_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        }

def save_RL_reduced_dataset(initial):
        df = pd.read_csv("data/"+initial+"_3t_data.csv", header = 0, \
        #df = pd.read_csv("RL_dataset/"+initial+"_raw.csv", header = 0, \
                names = ['rewards','cueballpos', 'cueballvel','redballpos', 'targetcornerpos', 'cueposfront', 'cueposback', 'cuedirection', 'cuevel'], usecols = [1,2,3,4,5,6,7,8,9], lineterminator = "\n")
        df = df.replace([r'\n', r'\[', r'\]'], '', regex=True) 
        rewards= pd.DataFrame.from_records(np.array(df['rewards'].astype(str).str.split(','))).astype(float)
        cueballpos = pd.DataFrame.from_records(np.array(df['cueballpos'].str.split(',')), columns=["trial","x","y","z"]).astype(float)
        cueballvel = pd.DataFrame.from_records(np.array(df['cueballvel'].str.split(',')), columns=["trial","x","y","z"]).astype(float)
        redballpos = pd.DataFrame.from_records(np.array(df['redballpos'].str.split(',')), columns=["trial","x","y","z"]).astype(float)
        targetcornerpos = pd.DataFrame.from_records(np.array(df['targetcornerpos'].str.split(',')), columns=["trial","x","y","z"]).astype(float)
        cueposfront = pd.DataFrame.from_records(np.array(df['cueposfront'].str.split(',')), columns=["trial","x","y","z"]).astype(float)
        cueposback = pd.DataFrame.from_records(np.array(df['cueposback'].str.split(',')), columns=["trial","x","y","z"]).astype(float)
        cuedirection = pd.DataFrame.from_records(np.array(df['cuedirection'].str.split(',')), columns=["trial","x","y","z"]).astype(float)
        cuevel = pd.DataFrame.from_records(np.array(df['cuevel'].str.split(',')), columns=["trial","x","y","z"]).astype(float)

        ball_data = {'cueballpos': cueballpos,
             'cueballvel': cueballvel,
            'redballpos': redballpos,
            'targetcornerpos': targetcornerpos
            }
        cue_data = {'cueposfront': cueposfront,
                'cueposback': cueposback,
                'cuedirection': cuedirection    #, 'cuevel': cuevel
                }
        dataset = Offline_reduced(ball_data, cue_data, cuevel, rewards[0],state_dim=21)
        #dataset = Offline_One(ball_data, cue_data, cuevel, rewards[0])  #rewars[0] to get float value from dataframe
        # Dataframe does not accept 2-d arrays
        # transform 2-d array (n states times 14 dimensions (state)) to 1-d array of list (of length 14)
        #dataset_RL = Offline_RL_dataset(data,rewards,cuedirection, cuevel,cueballpos, terminate_on_end=True)
        #Offline_RL_load(rewards,cueballpos,redballpos, targetcornerpos, cueposfront, cueposback, cuedirection, cuevel, terminate_on_end=True)
        #dataset = Offline_One_big_dict(data, initial)

        new_d = {'trial': dataset["trial"],
                'states': np.zeros(dataset["states"].shape[0], dtype=object),
                'actions': np.zeros(dataset["actions"].shape[0], dtype=object),
                'new_states': np.zeros(dataset["new_states"].shape[0], dtype=object),
                'rewards': dataset["rewards"],
                'terminals': dataset["terminals"]}
        for i in range(dataset["states"].shape[0]):
                new_d['states'][i] = dataset["states"][i][:].tolist()
                new_d['actions'][i] = dataset["actions"][i][:].tolist()
                new_d['new_states'][i] = dataset["new_states"][i][:].tolist()

        pd_dataset = pd.DataFrame.from_dict(new_d)
        pd_dataset.to_csv("data_original/"+initial+"_Offline_reduced.csv")
        #pd_dataset.to_csv("RL_dataset/"+initial+".csv")
        print(initial, "reduced Offline dataset saved")    
 #################################################################################################################
        
#################################################################################################################

def define_rewards(dataset):
    #reward_ = []
    mean_pocket = 104.94109114059208 #105.0736 
    lo_bound_angle = mean_pocket - 0.9208495903864685	#0.3106383 #= 104.7629617
    up_bound_angle = mean_pocket + 0.9208495903864685
    radius = 0.001  #0.0005

    #Pay attention if Force or delta Force
    lo_bound_force = 0.08   #0.6
    up_bound_force = 0.5    #2.3
    
    #cuevel = compute_cuevel(dataset["states"][[12,13,14]])
    #cue_acceleration = compute_acceleration(cuevel)
    
    terminals_ind = dataset["trial"][dataset["terminals"]==True]
    for i in dataset["trial"].unique():
        
        hit_ind = terminals_ind[terminals_ind == i].index.values -2 #40
        end_funnel = terminals_ind[terminals_ind == i].index.values #-30
        start_funnel = end_funnel-3 #25

        for j in dataset["trial"][dataset["trial"] == i].index.values:#range(start_funnel, end_funnel+1):
            if j != terminals_ind[terminals_ind == i].index.values:
                ## Angle funnel reward
                angle = compute_angle_for_funnel(dataset["states"][[18,20]].iloc[j+1])  #-compute_angle_for_funnel(dataset["states"][[18,20]].iloc[j])
                #if angle-dataset["actions"][0].iloc[j] > 0.01:
                    #print("angle and action different: ", angle, dataset["actions"][0].iloc[j], i, j)
                if angle >= lo_bound_angle and angle <= up_bound_angle:
                    dataset["rewards"].iloc[j] += 0.01

                if j >= start_funnel and j < end_funnel:
                    ## Hitting ball reward
                    #if j+1 > hit_ind - 2 and j+1 < hit_ind + 1:
                    if (dataset["states"][12].iloc[j+1]-dataset["states"][0].iloc[j+1])**2 + (dataset["states"][13].iloc[j+1]-dataset["states"][1].iloc[j+1])**2\
                        + (dataset["states"][14].iloc[j+1]-dataset["states"][2].iloc[j+1])**2 < radius: 
                        dataset["rewards"].iloc[j] += 0.03
                    
                    ## Angle funnel around hitting reward
                    if angle >= lo_bound_angle and angle <= up_bound_angle:
                        dataset["rewards"].iloc[j] += 0.4

                    ## Cue acceleration reward
                    if dataset["actions"][1].iloc[j] >= lo_bound_force and dataset["actions"][1].iloc[j] <= up_bound_force: 
                        dataset["rewards"].iloc[j] += 0.1
                    """else:
                        ## Angle funnel around hitting reward
                        if angle >= lo_bound_angle and angle <= up_bound_angle:
                            dataset["rewards"].iloc[j] += 0.03"""

                        ## Cue acceleration reward
                        #if np.sin(angle)*dataset["actions"][1].iloc[j+1] >= lo_bound_force and np.sin(angle)*dataset["actions"][1].iloc[j+1] <= up_bound_force: 
                            #dataset["rewards"].iloc[j] += 0.01
    return dataset

def reset_rewards2(rewards, terminals, trial):
    
    for i in range(len(rewards)):
        if rewards.iloc[i] == 100.0:
            rewards.iloc[i] = 1.0
        else:
            rewards.iloc[i] = 0.0
    """terminals_ind = trial[terminals==True]
    new_rewards = pd.DataFrame(np.zeros(len(rewards)))
    #N=len(trial.unique()

    for i, t_ind in enumerate(trial.unique()):
        ind = terminals_ind[terminals_ind == t_ind]
        if terminals.iloc[ind]:
            if rewards.iloc[ind] == 100.0:
                new_rewards.iloc[ind] = 1.0
            elif rewards.iloc[ind] == -10.0 :
                new_rewards.iloc[ind] = 0.0
        else:
            new_rewards.iloc[ind] = 0.0"""
        
        
    """if i == N-1:        #No last timestep due to new_states, so advance reward for the last trial 
            if terminals.iloc[i]:
                if rewards.iloc[i+1] == 100.0:
                    rewards.iloc[i] = 1.0
                elif rewards.iloc[i+1] == -10.0 :
                    rewards.iloc[i] = 0.0
            else:
                rewards.iloc[i] = 0.0
        else:
            if terminals.iloc[i]:
                if rewards.iloc[i] == 100.0:
                    rewards.iloc[i] = 1.0
                elif rewards.iloc[i] == -10.0 :
                    rewards.iloc[i] = 0.0
            else:
                rewards.iloc[i] = 0.0"""
    #new_rewards =new_rewards.drop(index=N)
    #terminals =terminals.drop(index=N)
    return rewards, terminals

def import_rewards(path2):  #path1, 
        """og_df = pd.read_csv(path1, header = 0, \
                names = ['rewards','cueballpos', 'cueballvel','redballpos', 'targetcornerpos', 'cueposfront', 'cueposback', 'cuedirection', 'cuevel'], usecols = [1,2,3,4,5,6,7,8,9], lineterminator = "\n")
        og_df = og_df.replace([r'\n', r'\[', r'\]'], '', regex=True) 
        rewards= pd.DataFrame.from_records(np.array(og_df['rewards'].astype(str).str.split(','))).astype(float)"""


        # Read RLdataset
        df = pd.read_csv(path2, header = 0, \
                    names = ['trial','states','actions','new_states','rewards','terminals'], usecols = [1,2,3,4,5,6], lineterminator = "\n")
        df = df.replace([r'\n', r'\[', r'\]'], '', regex=True) 
        states = pd.DataFrame.from_records(np.array(df['states'].str.split(','))).astype(float)
        actions = pd.DataFrame.from_records(np.array(df['actions'].str.split(','))).astype(float)
        trial = df['trial'].astype(int)
        rewards = df['rewards'].astype(float)
        terminals = df['terminals'].astype(int)
        dataset = {'trial': trial,
                'states': states,
                'actions': actions,       
                'rewards': rewards,
                'terminals': terminals}
        #print(dataset["states"].shape)
        #rewards, terminals = reset_rewards2(rewards.squeeze(), terminals, trial)
        #dataset["terminals"] = terminals
        #dataset["rewards"] = rewards
        dataset = define_rewards(dataset)
        
       # pd_df_ter = pd.DataFrame.from_dict(dataset["terminals"])
        #df['terminals'] = pd_df_ter
        pd_df_rew = pd.DataFrame.from_dict(dataset["rewards"])
        df['rewards'] = pd_df_rew
                    
        return df


def add_rewards(dataset, funnel_Subj, targetball_Subj):
    terminals_ind = dataset["trial"][dataset["terminals"]==True]
    for i, t_ind in enumerate(dataset["trial"].unique()):
        ind = terminals_ind[terminals_ind == t_ind]
        #if funnel_Subj.iloc[i].values:  #dataset["rewards"].iloc[ind].values != 1.0 and 
            #dataset["rewards"].iloc[ind.index.values] += 1.0
        if i>= len(targetball_Subj):
            print(len(targetball_Subj), i, t_ind)
        #if targetball_Subj.iloc[i].values:
        dataset["rewards"].iloc[ind.index.values] += targetball_Subj.iloc[i].values
    return dataset

def modify_rewards(path_Subj, funnel_Subj_path, targetball_Subj_path):
    # Read RLdataset
    df = pd.read_csv(path_Subj, header = 0, \
                names = ['trial','states','actions','new_states','rewards','terminals'], usecols = [1,2,3,4,5,6], lineterminator = "\n")
    df = df.replace([r'\n', r'\[', r'\]'], '', regex=True) 
    #states = pd.DataFrame.from_records(np.array(df['states'].str.split(','))).astype(float)
    #actions = pd.DataFrame.from_records(np.array(df['actions'].str.split(','))).astype(float)
    trial = df['trial'].astype(int)
    rewards = df['rewards'].astype(float)
    terminals = df['terminals'].astype(int)
    dataset = {'trial': trial,  #'states': states, 'actions': actions,       
            'rewards': rewards,
            'terminals': terminals}

    funnel_Subj = pd.read_csv(funnel_Subj_path, header = 0,\
        names = ['rewards'], usecols = [1], lineterminator = "\n")
    funnel_Subj = funnel_Subj.replace([r'\n', r'\[', r'\]'], '', regex=True) 
    
    targetball_Subj = pd.read_csv(targetball_Subj_path, header = 0, \
    names = ['rewards'], usecols = [1], lineterminator = "\n")
    targetball_Subj = targetball_Subj.replace([r'\n', r'\[', r'\]'], '', regex=True) 

    dataset = add_rewards(dataset, funnel_Subj, targetball_Subj)
    pd_df_rew = pd.DataFrame.from_dict(dataset['rewards'])
    df['rewards'] = pd_df_rew
    return df 

######################################################## Main ##############################################################3

if __name__ == "__main__":
    
    path1 = "D/Round1/"
    path2 = "D/Round2/" 
    pathlist = [path1, path2]
    reward_path = "D/Rewards/"
    data_path = "data_original/"
    save_data = "data_shaped/"
    #for path in pathlist:
    
    list_pathdir = sorted(os.listdir(data_path))
    for path in pathlist:
        for i, initial in enumerate(sorted(os.listdir(path))):
            pathSubj = path + str(initial)
            for fil in range(len(sorted(os.listdir(pathSubj + '/Game/')))):
                if sorted(os.listdir(pathSubj + '/Game/'))[fil].find("Block2") > -1:
                    blockFile = sorted(os.listdir(pathSubj + '/Game/'))[fil]

                    if blockFile.find('Reward') > -1:
                        if blockFile.find('Left') > -1:
                            funnel_Subj_path = reward_path  + str(initial) + "_funnel_rewards.csv"
                            targetball_Subj_path = reward_path  + str(initial) + "_targetball_rewards.csv"

                            path_Subj = data_path + str(initial) + "_Offline_3t_data.csv"  #reduced

                            df = modify_rewards(path_Subj, funnel_Subj_path, targetball_Subj_path)
                            
                            path2_save = save_data + str(initial) + "_Offline_reduced.csv"#reduced
                            df.to_csv(path2_save)
                            print("modify rewards file ", data_path,initial, "saved in ", save_data)

    """
    reduced_data = sorted(os.listdir("reduced_data/"))
    data_path = "data_delta_original/"
    save_data = "data_delta_shaped/"
    
    path = "D/Round1/" 
    list_pathdir = sorted(os.listdir(data_path))
    for i, initial in enumerate(sorted(os.listdir(path))):
        #if initial == "SF" or initial == "TA" or initial == "TH" or initial == "TS" or initial == "VB" or initial == "WZ" or initial == "YC" or initial == "YH":
        path1 = "data/"+reduced_data[i]
        path2_ = data_path+list_pathdir[i]
        df = import_rewards(path2_) #path1, 
        path2_save = save_data+list_pathdir[i]
        df.to_csv(path2_save)
        print("modify rewards file ", path2_save, "saved in ", save_data) """
    
    
    
    '''path = "D/Round1/"  #"/mnt/c/Users/dario/Documents/DARIO/ETUDES/ICL/code/data/Round 1/"
    path2 = "D/Round2/"  #"/mnt/c/Users/dario/Documents/DARIO/ETUDES/ICL/code/data/Round 2/"
    pathlist = [path, path2]

    for i, initial in enumerate(sorted(os.listdir(path))):
        pathSubj = path + str(initial)
        for fil in range(len(sorted(os.listdir(pathSubj + '/Game/')))):
            if sorted(os.listdir(pathSubj + '/Game/'))[fil].find("Block2") > -1:
                blockFile = sorted(os.listdir(pathSubj + '/Game/'))[fil]
                
                if blockFile.find('Reward') > -1:
                    #data = resultsMultipleSubjects(path, initial, 'reward', 'all')
                    #save_raw_data(data, initial)
        save_RL_reduced_dataset(initial)'''