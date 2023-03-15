import os
import numpy as np
import torch
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
import tqdm
import math
import csv
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_
import seaborn as sns
import matplotlib.pyplot as plt"""
################################################### Functions ########################################################################################3
def compute_actions_2D(cuevel, cueDirection, cueposfront, cueposback):
    cueAngle = compute_angle(cueDirection) 
    shotMagnitude = compute_impulseForce_2D(cuevel, cueDirection)  #min_max_standardisation(, min_val=0, max_val=3)
    actions = np.hstack((cueAngle.reshape(-1,1) , shotMagnitude.reshape(-1,1) ))  #min_max_standardisation_acc(cuevel[["x","y","z"]].to_numpy())))   #, cueacceleration)) #, normalize(cueacceleration))  
    return actions  

def compute_actions(cuevel, cueDirection, cueposfront, cueposback):
    cueAngle = compute_angle(cueDirection)  #min_max_standardisation(, min_val=45, max_val=135)
    
    #cueacceleration =  normalize(compute_acceleration(cuevel[["x","y","z"]].to_numpy()))      # min_max_standardisation() #, min_val=-140, max_val=140)
    shotMagnitude = compute_impulseForce(cuevel, cueDirection)  #min_max_standardisation(, min_val=0, max_val=3)
    actions = np.hstack((cueAngle.reshape(-1,1) , shotMagnitude.reshape(-1,1) ))  #min_max_standardisation_acc(cuevel[["x","y","z"]].to_numpy())))   #, cueacceleration)) #, normalize(cueacceleration))  
    #vel_x = min_max_standardisation(cuevel[["x"]].to_numpy())
    #vel_y = min_max_standardisation(cuevel[["y"]].to_numpy())
    #vel_z = min_max_standardisation(cuevel[["z"]].to_numpy())
    #actions = np.hstack((cueAngle.reshape(-1,1), cueposfront[["x", "y", "z"]].to_numpy(), cueposback[["x", "y", "z"]].to_numpy(), vel_x, vel_y, vel_z))  #, cuevel.to_numpy())
    return actions  #cueAngle, cueMagnitude

def compute_acceleration(cuevel):
    dt = 0.0111
    acc = np.zeros(cuevel.shape)
    for i in range(len(cuevel)):
        if i == 0:
            acc[i][:] = 0   #(np.abs(cuevel[i][:]-0))/dt
        else:
            acc[i][:] = (cuevel[i][:]-cuevel[i-1][:])/dt
            #if np.abs(acc[i][0]) > 10 or np.abs(acc[i][1]) > 10:
                #print("cuevel values for big acc: ", cuevel[i][:], cuevel[i-1][:], i)
    
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
    #if min_val == 0:
        #min_val = array.min()
        #max_val = array.max()
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

def compute_impulseForce_2D(cuevel, cuedirection):
    shotMagnitude = np.linalg.norm(cuevel[["x","z"]], axis=1)  
    return shotMagnitude    #, impulseForce

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

def def_reward(Angle_data): 
    reward = np.zeros(len(Angle_data.Trial))
    
    for i in range(len(Angle_data.Trial)):
        if Angle_data['Success'][i] == 1:
            reward[i] = 1.0
        elif Angle_data['SuccessFunnel'][i] == 1:
            reward[i] = 1.0
        #elif SuccessMedian_table.iloc[i] == 1:     #Agent will not understand reward for trial better than past 10 trials
            #reward[i] = 20
        else:
            reward[i] = 0.0

    return reward

def def_reward_funnel(Angle_data, start_ind, trial): #Success, SuccessFunnel_table, SuccessMedian_table):
    #print("Reward function")
    reward = np.zeros(len(start_ind))
    for i in range(len(start_ind)):
        if Angle_data['Success'].iloc[trial[i]-1] == 1:
            reward[i] = 1.0
        elif Angle_data['SuccessFunnel'].iloc[trial[i]-1] == 1:
            reward[i] = 1.0
        #elif SuccessMedian_table.iloc[i] == 1:     #Agent will not understand reward for trial better than past 10 trials
            #reward[i] = 20
        else:
            reward[i] = 0.0

    return reward


def def_target_ball_reward(cueball_pos, targetball_pos, targetball_vel, start_ind, trial):
    targetball_reward = np.zeros(len(start_ind))
    for i, index in enumerate(start_ind):
        count=index
        go_on = True
        while count < len(targetball_pos["trial"]) and targetball_pos["trial"].iloc[count] == trial[i] and go_on:
            if in_radius_around_cueball(cueball_pos.iloc[count], targetball_pos.iloc[count]):
                if np.linalg.norm(targetball_vel[["x","z"]].iloc[count]) > 0.01:
                    targetball_reward[i] = 0.3 
                    go_on=False
            count+=1

    return targetball_reward



def static_for_n_timesteps(cueballvel, redballvel, timestep, n):
    err = 0.001
    count=0
    for i in range(timestep, timestep+n, 1):
        if np.linalg.norm(cueballvel[["x", "y", "z"]].iloc[i].to_numpy()) < err and np.linalg.norm(redballvel[["x", "y", "z"]].iloc[i].to_numpy()) < err:
            count += 1
    if count == n:
        return True
    else:
        return False

def in_radius_around_cueball(cueballpos, cueposfront):
    radius = 0.1
    if np.linalg.norm(cueballpos[["x", "y", "z"]] - cueposfront[["x", "y", "z"]]) < radius:
        return True
    else:
        return False
def start_hit_reduced_timesteps(subjData):
    
    print("start hit timesteps function")
    cueballPos = subjData["cbpos"]
    cueballvel = subjData["cbvel"]
    redballvel = subjData["rbvel"]
    cuevel = subjData["cuevel"]
    #Find when cue hits cueball
    hit_ind=[]
    start_ind=[]
    n = 9
    threshold = 0.06

    start_movement = False

    prev_trial = 0
    start_movement_ind = 0
    window_before_hit = 3   #100  #200
    window_after_hit = 2    #40   #50
    count_miss = 1

    for i in range(len(cueballPos["trial"])):
        #The first 200 timesteps approximately are calibration and parasite movements
        if i > 200:  
            #New trial started
            if cueballPos["trial"].iloc[i] > prev_trial:
                if cueballPos["trial"].iloc[i]== (len(start_ind) - count_miss):
                    print("WARNING: no start and hit index defined for trial ", prev_trial)
                    count_miss += 1
                if static_for_n_timesteps(cueballvel, redballvel, i, n):    #and in_radius_around_cueball(cueballpos.iloc[i].to_numpy(), cueposfront.iloc[i].to_numpy()):
                    #Wait for cueball vel y-axis and redball vel y-axis to be zero after new trial started
                    start_movement_ind = i
                    start_movement = True
                    prev_trial = cueballPos["trial"].iloc[i]

            elif cueballPos["trial"].iloc[i] == prev_trial and start_movement == True:        
                if np.linalg.norm(cueballvel[["x", "z"]].iloc[i].to_numpy()) > threshold:
                    if len(start_ind) > 0:
                        if cueballPos["trial"][i-window_before_hit] != cueballPos["trial"][start_ind[-1]]:
                            start_ind = np.append(start_ind, i-window_before_hit)
                        else:
                            start_ind = np.append(start_ind, start_movement_ind)
                    else:
                        start_ind = np.append(start_ind, i-window_before_hit)
                    #if cueballPos["trial"][start_ind[i]] == cueballPos["trial"][start_ind[i+1]]:
                        #print("same trial 2 index", start_movement, cueballPos["trial"].iloc[i], prev_trial)
                    hit_ind = np.append(hit_ind, i+window_after_hit)   #+1 when selecting from trajectory
                    start_movement = False
                        
    if len(start_ind) < 250 or len(hit_ind) < 250:
        print("WARNING: either missed a start-hit index, or a trial was not properly recorded")
    for i in range(len(start_ind)-1):
        if cueballPos["trial"][start_ind[i]] == cueballPos["trial"][start_ind[i+1]]:
            print("start index ", i, " and start index ", i+1, "have same trial number")
        if start_ind[i] >= hit_ind[i]:
            raise ValueError("start ind > hit_ind", i , start_ind[i], hit_ind[i])
    return start_ind, hit_ind#.astype(int)


#######################################################################################################################################################################

# Reward Task
def importDataReward (initials, path, pocket, lbThreshold = -9.364594, ubThreshold = 4.675092):

    '''Imports Behavioural Data for a subject performing Reward task
    Args:
    - initials (str): initials of the subject
    - path (str): path of the data
    - pocket (str): left/right corresponding to the pocket
    - lbThreshold (float), ubThreshold (float): 
        lower and upper bounds of the interval of possible angles (out of the interval it will be considered outlier)
    
    Output:
    - rewardSubjectData (dict): combinations initials (key) - dataframe (value) for each subject with chosen pocket for reward task
        Every entry of the dictionary stores four data frames:
            1. VRData: behavioural data from the unity game
            2. angleData: angle of the shot and corresponding trial numbers (250x2)
            3. successData: binary variable for success and corresponding trial numbers (250x2)
            4. rewardMotivation: reward motivation (Funnel or Median) and corresponding trial numbers (250x2)
    
    '''

    # Declare possible pocket and raise error otherwise
    pockets = ['left', 'right']
    if pocket.lower() in pockets:
        pocket = pocket.capitalize()
    else:
        raise ValueError("Invalid game type. Expected one of: %s" % pockets)

    ### Definition of ideal angle and funnel ###
    idealAngle = 105.0736
    lb_idealAngle = 104.5401 - idealAngle
    ub_idealAngle = 105.6072 - idealAngle

    ### Import angle data ###
    angleData = pd.read_csv(path + initials + "/Game/" + initials + "_Angle.txt", header=None, names = ['Angle'])
    angleData['Block'] = np.repeat(range(1,11),25)
    # Remove outliers setting values to nan
    angleData.loc[(angleData.Angle - idealAngle) > ubThreshold, 'Angle'] = np.nan
    angleData.loc[(angleData.Angle - idealAngle) < lbThreshold, 'Angle'] = np.nan   

    ### Import success data ###
    successData = pd.read_csv(path + initials + "/Game/" + initials + "_Success.txt",  sep = '\t', header = None, names = ['Block','Trial','Angle','Magnitude','RBPosition'], usecols = [0,1,2,3,5])

    ### Import Reward Motivation (funnel or improvement) ###
    rewardMotivation = pd.read_csv(path + initials + "/Game/" + initials + "_RewardMotivation.txt",  sep = '\t', header = None, names = ['Block', 'Trial','Motivation'])
    rewardMotivation['Trial'] = (rewardMotivation['Block']-1)*25 + rewardMotivation['Trial']


    ### Import Behavioural Data from Blocks ###
    VRData = pd.DataFrame()

    for bl in range(1,11):
        block = pd.read_csv(path + initials + "/Game/" + initials + "_Reward" + pocket + "_Block" + str(bl) + ".txt", sep = '\t')
        while block['TrialNumber'].iloc[-1]!=25:
            block.drop(block.tail(1).index,inplace=True)
        if block['TrialNumber'].iloc[-1]==26:
            print("block: ", bl,block.tail(1).index)
        block['TrialNumber'] = block['TrialNumber'] + (bl-1)*25  
        VRData = pd.concat([VRData, block])
        #print("Block" + str(bl))

    
    ### Store data in a dictionary ###
    rewardSubjectData = {}
    names = ['VRData','Angle','Success','Motivation']
    dfs = [VRData, angleData, successData, rewardMotivation]
    counter = 0

    for df in names:
        rewardSubjectData[df] = dfs[counter]
        counter += 1

    ### Return the dictionary ###    
    return(rewardSubjectData)


def preprocess(dataset):

    '''Preprocess data from Pool VR returning position of cue ball, red ball, stick and gaze
    
    Args:
    - dataset (pd.Dataframe): VRData (1st dictionary entry of the output of importDataError or importDataReward)
    
    Output:
    - subject (dict): four dataframes for one single subject
        1. cbpos: position of the cue ball (X, Y, Z), trial number, time (in the game) and real time (N x 6)
        2. rbpos: position of the red ball, trial number, time (in the game) and real time (N x 6)
        3. stick: position of the stick, trial number, time (in the game) and real time (N x 6)
        4. gaze: position of the gaze, trial number, time (in the game) and real time (N x 6)
    
    '''

    ### Create string variables from 3D vectors ###
    
    dataset['cueballpos_str'] = dataset['cueballpos'].str.split(',')
    dataset['cueballvel_str'] = dataset['cueballvel'].str.split(',')
    dataset['redballpos_str'] = dataset['redballpos'].str.split(',')
    dataset['redballvel_str'] = dataset['redballvel'].str.split(',')
    dataset['cueposfront_str'] = dataset['cueposfront'].str.split(',')
    dataset['cueposback_str'] = dataset['cueposback'].str.split(',')
    dataset['cuevel_str'] = dataset['cuevel'].str.split(',')
    dataset['cuedirection_str'] = dataset['cuedirection'].str.split(',')
    dataset['corner1pos_str'] = dataset['corner1pos'].str.split(',')
    dataset['corner5pos_str'] = dataset['corner5pos'].str.split(',')
    dataset['corner6pos_str'] = dataset['corner6pos'].str.split(',')
    
    dataset['stick'] = dataset['optifront'].str.split(',')
    dataset['gaze_str'] = dataset['gaze'].str.split(',')

    ### Create datasets from string variables ###
    cbpos = pd.DataFrame.from_records(np.array(dataset['cueballpos_str']), columns=['x','y','z']).astype(float)
    cbvel = pd.DataFrame.from_records(np.array(dataset['cueballvel_str']), columns=['x','y','z']).astype(float)
    rbpos = pd.DataFrame.from_records(np.array(dataset['redballpos_str']), columns=['x','y','z']).astype(float)
    rbvel = pd.DataFrame.from_records(np.array(dataset['redballvel_str']), columns=['x','y','z']).astype(float)
    cueposfront = pd.DataFrame.from_records(np.array(dataset['cueposfront_str']), columns=['x','y','z']).astype(float)
    cueposback = pd.DataFrame.from_records(np.array(dataset['cueposback_str']), columns=['x','y','z']).astype(float)
    cuedirection = pd.DataFrame.from_records(np.array(dataset['cuedirection_str']), columns=['x','y','z']).astype(float)
    cuevel = pd.DataFrame.from_records(np.array(dataset['cuevel_str']), columns=['x','y','z']).astype(float)
    corner1pos = pd.DataFrame.from_records(np.array(dataset['corner1pos_str']), columns=['x','y','z']).astype(float)
    corner5pos = pd.DataFrame.from_records(np.array(dataset['corner5pos_str']), columns=['x','y','z']).astype(float)
    corner6pos = pd.DataFrame.from_records(np.array(dataset['corner6pos_str']), columns=['x','y','z']).astype(float)

    stick = pd.DataFrame.from_records(np.array(dataset['stick']), columns=['x','y','z']).astype(float)
    gaze = pd.DataFrame.from_records(np.array(dataset['gaze_str']), columns=['x','y','z']).astype(float)


    ### Standardise w.r.t cue ball initial position and add time and trial number ###
    #x_std, y_std, z_std = cbpos.iloc[0]
    #d = 0.2 #offset, see cs Script
    #x_ref, y_ref, z_ref = corner1pos.iloc[100]+d  #to be sure position has been calibrated
    #x_scale, y_scale, z_scale = np.abs((corner1pos.iloc[100]-d)-(corner6pos.iloc[100]+d))
    #print("scaling params: ", (x_ref, y_ref, z_ref), (x_scale, y_scale, z_scale))
    for df in (cbpos, rbpos, cueposfront, cueposback, corner5pos, corner6pos, stick, gaze):    # cbvel, rbvel, cuedirection, cuevel, 
        #df -= (x_ref, y_ref, z_ref)
        #df[["x", "z"]] /= (x_scale, z_scale)
        df['trial'] = np.array(dataset['TrialNumber'])
        #df['time'] = np.array(dataset['TrialTime']) 
        #df['timeReal'] = np.array(dataset['GlobalTime'])  
    
    #We don't want to standardize the velocities with respect to cue ball initial position
    for df in (cbvel, rbvel, cuedirection, cuevel):
        df['trial'] = np.array(dataset['TrialNumber'])

    ### Create a dictionary for saving dataframes and save them ###
    subject = {}
    names = ['cbpos', 'cbvel', 'rbpos', 'rbvel', 'cueposfront', 'cueposback', 'cuedirection', 'cuevel', 'corner5pos', 'corner6pos', 'stick','gaze']
    dfs = [cbpos, cbvel, rbpos, rbvel, cueposfront, cueposback, cuedirection, cuevel, corner5pos, corner6pos, stick, gaze]
    counter = 0

    for df in names:
        subject[df] = dfs[counter]
        counter += 1

    ### Return the dictionary ###
    return(subject)


def successTrialsReward(subjData, preprocessData):

  '''Preprocesses and derive successful trials from VR Data correcting the output for reward subjects, \
    creating three new variables in the Angle dataset: SuccessMedian (binary variable for success attributable to the median), \
    SuccessFunnel (binary variable for success attributable to the funnel) and Target (reference angle to be rewarded)
  
    Args:
    - subject (dict): output of importDataError or importDataReward
    
    Output:
    - subject (dict): copy of the input with correct success definition and three new variables in the Angle dataset
   
  '''
  
  print("Success Derivation function")
  subject = subjData
  ### Set threshold for funnel ###
  idealAngle = 105.0736
  lb_idealAngle = 104.5401 - idealAngle
  ub_idealAngle = 105.6072 - idealAngle

  ### Derive Successful trials from Angle dataset ###
  ind = subject['Angle']['Block'].isin([1,2,3,10])

  ### Create new variables in Angle dataset: success overall, success due to funnel and success due to improvement ###
  subject['Angle']['Success'] = 0
  subject['Angle']['SuccessFunnel'] = 0
  subject['Angle']['SuccessMedian'] = 0

  ### Add success to angle dataset ###
  subject['Angle']['Success'][ind] = subject['Angle'].Angle[ind].isin(subject['Success']['Angle']).astype(int)

  ### Add trial and standardised angle ###
  subject['Angle']['Trial'] = range(1,251)
  subject['Angle']['AngleStd'] =  subject['Angle']['Angle'] - idealAngle



  ### Preprocess game data ###
  #gameData = preprocess(subject['VRData'])

  ### True Success for baseline blocks ###
  succTrials = subject['Angle'].Success.index[subject['Angle'].Success == 1] + 1
  
  for tr in succTrials:
      cbTrial = preprocessData['cbpos'][preprocessData['cbpos'].trial == tr]

      if (cbTrial['y'].iloc[0] - cbTrial['y'].iloc[-1]) >= 0.02:
          subject['Angle'].Success.iloc[tr-1] = 0

  print(str(succTrials.shape[0]-sum(subject['Angle'].Success)) + " fake successes removed")

  ### Success for perturbation blocks ###
  for tr in subject['Motivation'].Trial:
    #subject['Angle']['Success'].iloc[tr] = 1

    if (subject['Motivation'].Motivation[subject['Motivation'].Trial == tr] == 'Median').all():
      subject['Angle']['SuccessMedian'][tr] = 1
    else:
      subject['Angle']['SuccessFunnel'][tr] = 1 

  ### Derive target for reward (median of the past 10 successful trials) ###

  target = []
  vec_median = list(subject['Angle']['AngleStd'].iloc[range(66,76)])

  for tr in range(76, 226):
    if subject['Angle']['Success'].iloc[tr]==1:
        target.append(min(ub_idealAngle, max(np.median(vec_median), ub_idealAngle-5)))
        vec_median.remove(max(vec_median))
        vec_median.append(subject['Angle']['AngleStd'].iloc[tr])
    else:
        target.append(min(ub_idealAngle, max(np.median(vec_median), ub_idealAngle-5)))

  target_overall = pd.Series(np.concatenate([np.repeat(np.nan, 75), target, np.repeat(np.nan, 25)]))

  subject['Angle']['Target'] = target_overall
  ### Returns preprocessed data with correct success ###
  return(subject)

def resultsSingleSubject(initials, gameType, path, pocket):
    '''
    Produces results for single subject starting from raw data

    Args:
    - initials (str): initials of the subject
    - gameType (str): error/reward - game mode
    - path (str): path of the data
    - pocket (str): left/right - corresponding to the pocket
    
    Output:
    - outputDict (dict): dictionary with two entries:
        1. Game (dict): game data (same output as importDataError/importDataReward) with success corrected
            - VRData: data from the Unity Game
            - Angle: shot directional angles for all trials
            - Success: features of all successful trials
            - Motivation (only for reward task): reason of successful trial (if funnel or improvement of the median)

        2. PreProcessed (dict): pre-processed game data
            - cbpos: position of the cue ball for all timeframes and all trials
            - rbpos: position of the red ball for all timeframes and all trials
            - stick: position of the stick for all timeframes and all trials
            - gaze: position of the gaze


    '''
    
    print("Result Single Subject function")
    # Declare possible gameType and raise error otherwise
    game_types = ['error', 'reward']
    if gameType.lower() in game_types:
        mode = gameType.lower()
    else:
        raise ValueError("Invalid game type. Expected one of: %s" % game_types)



    ##### Error Mode #####
    if mode == 'error':

        # Import Data
        gameData = importDataError(initials, path, pocket) 
        ''' 
        gameData is a dictionary with keys:
        - VRData: data from the Unity Game
        - Angle: shot directional angles for all trials
        - Success: features of all successful trials
        '''

        # Preprocess Data
        preprocData = preprocess(gameData['VRData'])
        '''
        preprocData is a dictionary with keys:
        - cbpos: position of the cue ball for all timeframes and all trials
        - rbpos: position of the red ball for all timeframes and all trials
        - stick: position of the stick for all timeframes and all trials
        - gaze: position of the gaze
        '''

        # Update of Success in Import Data
        gameData = successTrialsError(gameData)



    ##### Reward Mode #####

    elif mode == 'reward':

        # Import Data
        gameData = importDataReward(initials, path, pocket) 
        ''' 
        gameData is a dictionary with keys:
        - VRData: data from the Unity Game
        - Angle: shot directional angles for all trials
        - Success: features of all successful trials
        - Motivation: reason of successful trial (if funnel or improvement of the median)
        '''

        # Preprocess Data
        preprocData = preprocess(gameData['VRData'])
        '''
        preprocData is a dictionary with keys:
        - cbpos: position of the cue ball for all timeframes and all trials
        - rbpos: position of the red ball for all timeframes and all trials
        - stick: position of the stick for all timeframes and all trials
        - gaze: position of the gaze
        '''

        # Update of Success in Import Data
        angleData = successTrialsReward(gameData, preprocData)


    ### List of outputs ###
    outputDict = {}
    outputDict['Game'] = gameData 
    outputDict['PreProcessed'] = preprocData
    outputDict['Angle'] = angleData
    return(outputDict)

def get_subjects_reward(path, sub, gameType, pocket):
    print(path, sub)
    # Derive path for a specific subject and game type
    pathSubj = path + str(sub)
    for fil in range(len(sorted(os.listdir(pathSubj + '/Game/')))):
        if sorted(os.listdir(pathSubj + '/Game/'))[fil].find("Block2") > -1:
            blockFile = sorted(os.listdir(pathSubj + '/Game/'))[fil]

    if (blockFile.find('Left') > -1 and pocket != 'right'):
        pocketSub = 'Left'
    elif (blockFile.find('Right') > -1 and pocket != 'left'):
        pocketSub = 'Right'
    else:
        pocketSub = 'None'
    # Call dataframe for both rounds together
    dataset = {}
    if blockFile.find('Reward') > -1:
        # Derive initials of the subject
        initials = os.path.basename(pathSubj)

        # Derive All Results for One Subject
        subjData = resultsSingleSubject(initials, 'reward', path, pocketSub)

        # Add data to dataframes previously defined
        dataset[str(initials)]={}

        
        dataset[str(initials)]["start_ind"], dataset[str(initials)]["hit_ind"] = start_hit_reduced_timesteps(subjData["PreProcessed"])  # start_hit_timesteps(subjData["PreProcessed"])     
        dataset[str(initials)]["start_ind"] = dataset[str(initials)]["start_ind"].astype(int)

        trial = subjData["PreProcessed"]["cbpos"]["trial"].iloc[dataset[str(initials)]["start_ind"]].values
        targetball_reward = def_target_ball_reward(subjData["PreProcessed"]["cbpos"], subjData["PreProcessed"]["rbpos"], subjData["PreProcessed"]["rbvel"], dataset[str(initials)]["start_ind"], trial)
        funnel_reward = def_reward_funnel(subjData["Angle"]["Angle"], dataset[str(initials)]["start_ind"], trial)
        
    return funnel_reward, targetball_reward

def resultsMultipleSubjects(path, sub, gameType, pocketSide):
    '''
    Args:
    - pathList (list): list of all paths of the folder with raw data
    - gameType (str): error/reward - game type 
    - pocketSide (str): left/right - game pocket
 
    '''
	
    print("result Multiple Subject function")
    # Declare possible gameType and raise error otherwise
    game_types = ['error', 'reward']
    if gameType.lower() in game_types:
        mode = gameType.lower()
    else:
        raise ValueError("Invalid game type. Expected one of: %s" % game_types)

    # Call dataframe for both rounds together
    dataset = {}
    
    # Declare possible pocketSide and raise error otherwise
    pocketChoice = ['left', 'right', 'all']
    if pocketSide.lower() in pocketChoice:
        pocket = pocketSide.lower()
    else:
        raise ValueError("Invalid pocket. Expected one of: %s" % pocketChoice)


    print(path, sub)
    if mode == 'reward':

        # Derive path for a specific subject and game type
        pathSubj = path + str(sub)
        for fil in range(len(sorted(os.listdir(pathSubj + '/Game/')))):
            if sorted(os.listdir(pathSubj + '/Game/'))[fil].find("Block2") > -1:
                blockFile = sorted(os.listdir(pathSubj + '/Game/'))[fil]


        if (blockFile.find('Left') > -1 and pocket != 'right'):
            pocketSub = 'Left'
        elif (blockFile.find('Right') > -1 and pocket != 'left'):
            pocketSub = 'Right'
        else:
            pocketSub = 'None'

        if blockFile.find('Reward') > -1:
            # Derive initials of the subject
            initials = os.path.basename(pathSubj)

            # Derive All Results for One Subject
            subjData = resultsSingleSubject(initials, 'reward', path, pocketSub)

            # Add data to dataframes previously defined
            dataset[str(initials)]={}

            dataset[str(initials)]["rewards"] = def_reward(subjData["Angle"]["Angle"])  #["Success"], subjData["Angle"]["Angle"]["SuccessMedian"], subjData["Angle"]["Angle"]["SuccessFunnel"])


            dataset[str(initials)]["cueballpos"] = subjData["PreProcessed"]["cbpos"]
            dataset[str(initials)]["cueballvel"] = subjData["PreProcessed"]["cbvel"]
            
            dataset[str(initials)]["start_ind"], dataset[str(initials)]["hit_ind"] = start_hit_reduced_timesteps(subjData["PreProcessed"])  # start_hit_timesteps(subjData["PreProcessed"])     
            dataset[str(initials)]["start_ind"] = dataset[str(initials)]["start_ind"].astype(int)
            dataset[str(initials)]["hit_ind"] = dataset[str(initials)]["hit_ind"].astype(int)
            dataset[str(initials)]["redballpos"] = subjData["PreProcessed"]["rbpos"]
            
            if pocketSub == 'Left':
                dataset[str(initials)]["targetcornerpos"] = subjData["PreProcessed"]["corner5pos"]
            else:
                dataset[str(initials)]["targetcornerpos"] = subjData["PreProcessed"]["corner6pos"]

            dataset[str(initials)]["cueposfront"] = subjData["PreProcessed"]["cueposfront"]

            dataset[str(initials)]["cueposback"] = subjData["PreProcessed"]["cueposback"]

            dataset[str(initials)]["cuedirection"] = subjData["PreProcessed"]["cuedirection"]

            dataset[str(initials)]["cuevel"] = subjData["PreProcessed"]["cuevel"]

            #dataset[str(initials)]["rewards"] = def_reward(subjData['Game']["Angle"]["Success"], subjData["Game"]["Angle"]["SuccessFunnel"], subjData["Game"]["Angle"]["SuccessMedian"])

            print("Imported " + initials + " as reward subject " + "for " + pocketSub + " pocket" + " in round: " + path)
            
    return dataset

def Offline_reduced_2D(data, cue_data, cuevel, rewards, state_dim=14, action_dim=2):   #state_dim should be a multiple of 6
    
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
    actions = compute_actions_2D(cuevel, cue_data["cuedirection"], cue_data["cueposfront"], cue_data["cueposback"])
   
    for i in range(N):
        count=0
        #count_action=0
        for x in data:
            state[count] = data[str(x)]["x"].iloc[i]
            #state[count+1] = data[str(x)]["y"].iloc[i]
            state[count+1] = data[str(x)]["z"].iloc[i]
            new_state[count] = data[str(x)]["x"].iloc[i+1]
            #new_state[count+1] = data[str(x)]["y"].iloc[i+1]
            new_state[count+1] = data[str(x)]["z"].iloc[i+1]
            count+=2

        #Add last j timesteps of the cuepos to states
        #3 observations on x and z axis makes 6 observations
        #for j in range(cue_states_iterations): #Warning if state_dim not multiple of 6, division fail for loop
        for x in cue_data:
            state[count] = cue_data[str(x)]["x"].iloc[i]
            #state[count+1] = cue_data[str(x)]["y"].iloc[i]
            state[count+1] = cue_data[str(x)]["z"].iloc[i]
            new_state[count] = cue_data[str(x)]["x"].iloc[i+1]
            #new_state[count+1] = cue_data[str(x)]["y"].iloc[i+1]
            new_state[count+1] = cue_data[str(x)]["z"].iloc[i+1]
            '''if x == "cueposfront" or x == "cueposback" or x == "cuevel":
                action[count_action] = cue_data[str(x)]["x"].iloc[i+1]
                action[count_action+1] = cue_data[str(x)]["y"].iloc[i+1]
                action[count_action+2] = cue_data[str(x)]["z"].iloc[i+1]
                count_action+=3'''
            count+=2
            
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

def Offline_reduced(data, cue_data, cuevel, rewards, state_dim=21, action_dim=2):   #state_dim should be a multiple of 6
    
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


def save_raw_data(data, initial, save_dir):
    reward_ = []
    #rewards = []
    cueballpos = []
    cueballvel = []
    redballpos = []
    targetcornerpos = []
    cueposfront = []
    cueposback = []
    cuedirection = []
    cuevel = []
    radius = 0.005     #real value without margin=0.00027173 #radius is 0.625cm corresponds to around 0.00027173 in VR coordinates
    vertical_margin = 0.01
    total_len_trajectories = 0
    
    k=0
    for i in range(data[initial]["start_ind"].shape[0]):
        #reward_ = []
        trial = data[initial]["cueballpos"]["trial"][data[initial]["start_ind"][i]]
        for j in range(data[initial]["start_ind"][i], data[initial]["hit_ind"][i]):
            total_len_trajectories += 1
            if j == data[initial]["hit_ind"][i]-1:
                #if i == 12 :
                    #print("right hit index wtf is going on", j, k)
                #done_bool = True
                reward = data[initial]["rewards"][trial-1]
            else:
                #if i == 12:
                    #print("wrong hit index", j, k)
                #done_bool = False
                ## Discounted reward ##
                #gamma = 0.99 
                #reward = gamma**(j - data[initial]["hit_ind"][i]) * data[initial]["rewards"][i]
                #if (data[initial]["cueposfront"]["x"][j]-data[initial]["cueballpos"]["x"][j])**2 + (data[initial]["cueposfront"]["z"][j]-data[initial]["cueballpos"]["z"][j])**2 < radius \
                    #and data[initial]["cueposfront"]["y"][j] < data[initial]["cueballpos"]["y"][j] + vertical_margin:   #cueball radius
                        #reward = 5
                        #print("cue stick touching cueball " , j)
                #else:
                reward = 0.0
            #if i == 12:
                #print(reward)
            #+=1
            reward_.append(reward)
        #rewards.append(np.array(reward_))
        cueballpos.append(np.array((data[initial]["cueballpos"]["trial"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["cueballpos"]["x"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["cueballpos"]["y"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["cueballpos"]["z"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]])))
        cueballvel.append(np.array((data[initial]["cueballvel"]["trial"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["cueballvel"]["x"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["cueballvel"]["y"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["cueballvel"]["z"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]])))
        redballpos.append(np.array((data[initial]["redballpos"]["trial"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["redballpos"]["x"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["redballpos"]["y"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["redballpos"]["z"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]])))
        targetcornerpos.append(np.array((data[initial]["targetcornerpos"]["trial"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["targetcornerpos"]["x"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["targetcornerpos"]["y"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["targetcornerpos"]["z"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]])))
        cueposfront.append(np.array((data[initial]["cueposfront"]["trial"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["cueposfront"]["x"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["cueposfront"]["y"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["cueposfront"]["z"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]])))
        cueposback.append(np.array((data[initial]["cueposback"]["trial"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["cueposback"]["x"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["cueposback"]["y"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["cueposback"]["z"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]])))
        cuedirection.append(np.array((data[initial]["cuedirection"]["trial"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["cuedirection"]["x"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["cuedirection"]["y"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["cuedirection"]["z"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]])))
        cuevel.append(np.array((data[initial]["cuevel"]["trial"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["cuevel"]["x"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["cuevel"]["y"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]], data[initial]["cuevel"]["z"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]])))

    #for i in range(len(reward_)):
        #print(reward_[i])
    dic = {'rewards': np.array(reward_),    #rewards,  
    'cueballpos': np.zeros(total_len_trajectories, dtype=object),
    'cueballvel': np.zeros(total_len_trajectories, dtype=object),
    'redballpos': np.zeros(total_len_trajectories, dtype=object),
    'targetcornerpos': np.zeros(total_len_trajectories, dtype=object),
    'cueposfront': np.zeros(total_len_trajectories, dtype=object),
    'cueposback': np.zeros(total_len_trajectories, dtype=object),
    'cuedirection': np.zeros(total_len_trajectories, dtype=object),
    'cuevel': np.zeros(total_len_trajectories, dtype=object)}

    transition_num = 0
    #print("cuballpos shape 1: ", cueballpos[1].shape[1], cueballpos[1].shape, data[initial]["start_ind"][1]-data[initial]["hit_ind"][1], rewards[1].shape)
    for i in range(data[initial]["start_ind"].shape[0]):
            for j in range(cueballpos[i].shape[1]):
                    dic['cueballpos'][transition_num] = [cueballpos[i][0][j],cueballpos[i][1][j], cueballpos[i][2][j], cueballpos[i][3][j]]
                    dic['cueballvel'][transition_num] = [cueballvel[i][0][j],cueballvel[i][1][j], cueballvel[i][2][j], cueballvel[i][3][j]]
                    dic['redballpos'][transition_num] = [redballpos[i][0][j], redballpos[i][1][j], redballpos[i][2][j], redballpos[i][3][j]]
                    dic['targetcornerpos'][transition_num] = [targetcornerpos[i][0][j], targetcornerpos[i][1][j], targetcornerpos[i][2][j], targetcornerpos[i][3][j]]
                    dic['cueposfront'][transition_num] = [cueposfront[i][0][j], cueposfront[i][1][j], cueposfront[i][2][j], cueposfront[i][3][j]]
                    dic['cueposback'][transition_num] = [cueposback[i][0][j], cueposback[i][1][j], cueposback[i][2][j], cueposback[i][3][j]]
                    dic['cuedirection'][transition_num] = [cuedirection[i][0][j], cuedirection[i][1][j], cuedirection[i][2][j], cuedirection[i][3][j]]
                    dic['cuevel'][transition_num] = [cuevel[i][0][j], cuevel[i][1][j], cuevel[i][2][j], cuevel[i][3][j]]
                    transition_num += 1
            #if i%50 == 0:
                    #print(i)
    pd_dataset = pd.DataFrame.from_dict(dic)
    pd_dataset.to_csv(save_dir+initial+"_3t_data.csv")
    #pd_dataset.to_csv("RL_dataset/raw_data/"+initial+"_raw_data.csv")
    #pd_dataset.to_csv("RL_dataset/"+initial+"_raw.csv")
    print("reduced raw data ", initial, " saved")
    #return pd_dataset, dic

def save_RL_reduced_dataset(initial, save_dir):
        df = pd.read_csv("data/"+initial+"_3t_data.csv", header = 0, #_3t_data\   
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
                'cuedirection': cuedirection,   #'cuevel': cuevel,
                }
        dataset = Offline_reduced(ball_data, cue_data, cuevel, rewards[0],state_dim=21)
        
        #dataset = Offline_reduced_2D(ball_data, cue_data, cuevel, rewards[0],state_dim=14)
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
        pd_dataset.to_csv(save_dir+initial+"_3t_left.csv")
        #pd_dataset.to_csv("RL_dataset/"+initial+".csv")
        print(initial, "reduced Offline dataset saved in ", save_dir)

if __name__ == "__main__":
    ############################# Define Path and Load Dataset ######################################

    path = "D/Round1/"  #"/mnt/c/Users/dario/Documents/DARIO/ETUDES/ICL/code/data/Round 1/"
    path2 = "D/Round2/"  #"/mnt/c/Users/dario/Documents/DARIO/ETUDES/ICL/code/data/Round 2/"
    pathlist = [path, path2]
    ## WARNING check which type of feedback in which round, and which corner for each subject
    #corner =  "left"
    #data = resultsSingleSubject(initial, "reward", path, corner)
    '''dic for one subject composed of ~1000 timepoints for one shot, 25 shots in one block, and 10 blocks
    First 3 blocks are baseline learning, then 6 blocks of adaptation to perturbation, and one final washout block
    That is 250 shots per subjects, 300'564 points in the dictionnary'''

    # Environment State Properties	
    for path in pathlist:			
        for i, initial in enumerate(sorted(os.listdir(path))):
            pathSubj = path + str(initial)
            for fil in range(len(sorted(os.listdir(pathSubj + '/Game/')))):
                if sorted(os.listdir(pathSubj + '/Game/'))[fil].find("Block2") > -1:
                    blockFile = sorted(os.listdir(pathSubj + '/Game/'))[fil]
                    
                    if blockFile.find('Reward') > -1:
                        if blockFile.find('Left') > -1:
                        #funnel_reward, targetball_reward = get_subjects_reward(path, initial, 'reward', 'all')

                        #pd.DataFrame(funnel_reward).to_csv("D/Rewards/"+initial+"_funnel_rewards.csv")
                        #pd.DataFrame(targetball_reward).to_csv("D/Rewards/"+initial+"_targetball_rewards.csv")
                        

                        #data = resultsMultipleSubjects(path, initial, 'reward', 'all')
                        #save_raw_data(data, initial, save_dir="data/")
                            save_RL_reduced_dataset(initial, save_dir="data_original_left/")
