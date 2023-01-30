import os
import numpy as np
import torch
import torch.nn as nn

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import scipy as sp

import itertools
import re

#import d4rl
import copy
import argparse
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################################################################################################################################
#                                                         Functions
########################################################################################################################################
def compute_impulseForce(cuevel, cuedirection):
    impulseForce = np.zeros(cuevel.shape)       #(N,2)
    shotMagnitude = np.zeros(1)
    shotDir = np.zeros(cuedirection.shape)
    #Reward: magnitude range
    lbMagnitude = 0.1   #0.516149
    ubMagnitude = 3 #0.882607

    shotMagnitude = np.sqrt(np.square(cuevel).sum(axis=1))
    #np.linalg.norm(cuevel, axis=1)
    for i in range(cuevel.shape[0]):
        if shotMagnitude[i] > ubMagnitude:
            shotMagnitude[i] = ubMagnitude
        elif shotMagnitude[i] < lbMagnitude:
            shotMagnitude[i] = 0

        shotDir[i][0] = cuedirection["x"].iloc[i]
        shotDir[i][1] = cuedirection["z"].iloc[i]
        if shotMagnitude[i] == 0:
            impulseForce[i][:] = 0
        else:
            impulseForce[i][:] = shotMagnitude[i] * shotDir[i][:]
    return impulseForce
    
def def_reward(SuccessFunnel_table, SuccessMedian_table):
    print("Reward function")
    reward = np.zeros(len(SuccessFunnel_table))
    for i in range(len(SuccessFunnel_table)):
        if SuccessFunnel_table.iloc[i] == 1:
            reward[i] = 100
        elif SuccessMedian_table.iloc[i] == 1:
            reward[i] = 20
        else:
            reward[i] = -10

    return reward

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

def start_hit_reduced_timesteps(subjData):
    
    print("start hit timesteps function")
    cueballPos = subjData["cbpos"]
    cueballvel = subjData["cbvel"]
    redballvel = subjData["rbvel"]
    cuevel = subjData["cuevel"]
    #Find when cue hits cueball
    hit_ind=[]
    start_ind=[]
    n = 10
    threshold = 0.06

    start_movement = False

    prev_trial = 0
    for i in range(len(cueballPos["trial"])):
        #The first 200 timesteps approximately are calibration and parasite movements
        if i > 200:   #!= 0:
            #New trial started
            if cueballPos["trial"].iloc[i] > prev_trial:
                if static_for_n_timesteps(cueballvel, redballvel, i, n):
                    #Wait for cueball vel y-axis and redball vel y-axis to be zero after new trial started
                    start_movement = True
                    prev_trial = cueballPos["trial"].iloc[i]
                    
            if start_movement == True and np.linalg.norm(cueballvel[["x", "z"]].iloc[i].to_numpy()) > threshold:
                start_ind = np.append(start_ind, i-30)
                hit_ind = np.append(hit_ind, i+9)   #+1 when selecting from trajectory
                start_movement = False

    if len(start_ind) != 250 or len(hit_ind) != 250:
        print("WARNING: either missed a start-hit index, or a trial was not properly recorded")
    for i in range(len(start_ind)):
        if start_ind[i] >= hit_ind[i]:
            raise ValueError("start ind > hit_ind", i , start_ind[i], hit_ind[i])
    return start_ind.astype(int), hit_ind.astype(int)

#################################### Import Data Functions ############################################

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


#################################### Preprocess ################################################

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
    corner5pos = pd.DataFrame.from_records(np.array(dataset['corner5pos_str']), columns=['x','y','z']).astype(float)
    corner6pos = pd.DataFrame.from_records(np.array(dataset['corner6pos_str']), columns=['x','y','z']).astype(float)

    stick = pd.DataFrame.from_records(np.array(dataset['stick']), columns=['x','y','z']).astype(float)
    gaze = pd.DataFrame.from_records(np.array(dataset['gaze_str']), columns=['x','y','z']).astype(float)


    ### Standardise w.r.t cue ball initial position and add time and trial number ###
    x_std, y_std, z_std = cbpos.iloc[0]
    for df in (cbpos, rbpos, cueposfront, cueposback, corner5pos, corner6pos, stick, gaze):    # cbvel, rbvel, cuedirection, cuevel, 
        df -= (x_std, y_std, z_std)

        df['trial'] = np.array(dataset['TrialNumber'])
        #df['time'] = np.array(dataset['TrialTime']) 
        #df['timeReal'] = np.array(dataset['GlobalTime'])  
    
    #We don't want to standardize the velocities with respect to cue ball initial position
    for df in (cbvel, rbvel, cuedirection, cuevel):
        df['trial'] = np.array(dataset['TrialNumber']).astype(int)

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

#################################### Succes Derivation ##########################################

def successTrialsReward(subject, preprocessData):

  '''Preprocesses and derive successful trials from VR Data correcting the output for reward subjects, \
    creating three new variables in the Angle dataset: SuccessMedian (binary variable for success attributable to the median), \
    SuccessFunnel (binary variable for success attributable to the funnel) and Target (reference angle to be rewarded)
  
    Args:
    - subject (dict): output of importDataError or importDataReward
    
    Output:
    - subject (dict): copy of the input with correct success definition and three new variables in the Angle dataset
   
  '''
  
  #print("Success Derivation function")

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

  #print(str(succTrials.shape[0]-sum(subject['Angle'].Success)) + " fake successes removed")

  ### Success for perturbation blocks ###
  for tr in subject['Motivation'].Trial:
    subject['Angle']['Success'].iloc[tr] = 1

    if (subject['Motivation'].Motivation[subject['Motivation'].Trial == tr] == 'Median').all():
      subject['Angle']['SuccessMedian'][tr] = 1
    else:
      subject['Angle']['SuccessFunnel'][tr] = 1 


  ### Derive target for reward (median of the past 10 successful trials) ###

  target = []
  vec_median = list(subject['Angle']['AngleStd'].iloc[range(66,76)][range(66,76)])

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

#################################### Results Single Subject #####################################

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

############################ Define Dataframe for all subjects ##################################
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


    #print(path, sub)
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

            dataset[str(initials)]["rewards"] = def_reward(subjData["Game"]["Angle"]["SuccessMedian"], subjData["Game"]["Angle"]["SuccessFunnel"])


            dataset[str(initials)]["cueballpos"] = subjData["PreProcessed"]["cbpos"]
            dataset[str(initials)]["cueballvel"] = subjData["PreProcessed"]["cbvel"]
            
            dataset[str(initials)]["start_ind"], dataset[str(initials)]["hit_ind"] = start_hit_reduced_timesteps(subjData["PreProcessed"])  #start_hit_timesteps(subjData["PreProcessed"])
        
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

def resultsMultipleSubjectsAllRound(pathList, gameType, pocketSide):

    ''' 
    Iterates resultsSingleSubject over all subjects of a path list for a specific game type for a specific pocket

    Args:
    - pathList (list): list of all paths of the folder with raw data
    - gameType (str): error/reward - game type 
    - pocketSide (str): left/right - game pocket


    Output:
    - error -> SubjError (dict): initials (str) - data (dict) combinations for all error subjects
    - reward -> SubjReward (dict): initials (str) - data (dict) combinations for all reward subjects

    '''

    # Declare possible gameType and raise error otherwise
    game_types = ['error', 'reward']
    if gameType.lower() in game_types:
        mode = gameType.lower()
    else:
        raise ValueError("Invalid game type. Expected one of: %s" % game_types)


    pocketChoice = ['left', 'right', 'all']
    if pocketSide.lower() in pocketChoice:
        pocket = pocketSide.lower()
    else:
        raise ValueError("Invalid pocket. Expected one of: %s" % pocketChoice)


    subjError = {}
    subjReward = {}


    for path in pathList:

        for sub in next(os.walk(path))[1]:

            if mode == 'error':

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

                if blockFile.find('Error') > -1:

                    if pocketSub == 'None':
                        continue
                    else:
                        # Derive initials of the subject
                        initials = os.path.basename(pathSubj)

                        # Derive All Results for One Subject
                        subjData = resultsSingleSubject(initials, 'error', path, pocketSub)
                        subjError[initials] = subjData

                        print("Imported " + initials + " as error subject " + "for " + pocketSub + " pocket")
            return(subjError)
        

        
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

                    if pocketSub == 'None':
                        continue
                    else:
                
                        # Derive initials of the subject
                        initials = os.path.basename(pathSubj)

                        # Derive All Results for One Subject
                        subjData = resultsSingleSubject(initials, 'reward', path, pocketSub)
                        subjReward[initials] = subjData

                        print("Imported " + initials + " as reward subject " + "for " + pocketSub + " pocket")
    return(subjReward)

############################## Build RL Offline Dataset #########################################

def Offline_Reduced(data, cue_data, cuevel, rewards, state_dim=14, action_dim=2):   #state_dim should be a multiple of 6
    
    print("Offline RL Dataset function")
    #number of trial
    N = len(data["cueballpos"]["trial"]) -1   #len(data[initial]["start_ind"])
    state_ = []
    new_state_ = []
    action_ = []
    reward_ = []
    done_ = []
    trial_ = []

    state = np.zeros(state_dim)  #np.zeros(21)
    new_state = np.zeros(state_dim)  #np.zeros(21)
    action = np.zeros(action_dim)
    #reward = np.zeros(1)
    #ball_states=6
    #cue_states=6
    #cue_states_iterations = int((state_dim-ball_states)/(cue_states))

    #Action Velocity, Force?
    #See car 2D -> action space acceleration and cueDirection
    actions = compute_impulseForce(cuevel[["x", "z"]], cue_data["cuedirection"][["x", "z"]])
        
    for i in range(N):
        count=0
        for x in data:
            state[count] = data[str(x)]["x"].iloc[i]
            state[count+1] = data[str(x)]["z"].iloc[i]
            new_state[count] = data[str(x)]["x"].iloc[i+1]
            new_state[count+1] = data[str(x)]["z"].iloc[i+1]
            count+=2

        #Add last j timesteps of the cuepos to states
        #3 observations on x and z axis makes 6 observations
        #for j in range(cue_states_iterations): #Warning if state_dim not multiple of 6, division fail for loop
        for x in cue_data:
            state[count] = cue_data[str(x)]["x"].iloc[i]#-j]
            state[count+1] = cue_data[str(x)]["z"].iloc[i]#-j]
            new_state[count] = cue_data[str(x)]["x"].iloc[i+1]#-j]
            new_state[count+1] = cue_data[str(x)]["z"].iloc[i+1]#-j]
            count+=2
   
        #Action Velocity, Force?
        action = actions[i][:]
        #reward = rewards[i]
        
        if data['cueballpos']['trial'].iloc[i+1] != data['cueballpos']['trial'].iloc[i] or i==N:
            done_bool = True
            reward = rewards.iloc[i]
        else:
            done_bool = False
            reward=0

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


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			print(next_state.size(), next_action.size())
			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

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
			pi = self.actor(state)
			Q = self.critic.Q1(state, pi)
			lmbda = self.alpha/Q.abs().mean().detach()

			actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action) 
			
			#print("iteration ", self.total_it, " actor_loss: ", actor_loss)

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

##################################################### Off-policy Evaluation ############################################################

class MAGIC(object):
    """Algorithm: MAGIC.
    """
    NUM_SUBSETS_FOR_CB_ESTIMATES = 25
    CONFIDENCE_INTERVAL = 0.9
    NUM_BOOTSTRAP_SAMPLES = 50
    BOOTSTRAP_SAMPLE_PCT = 0.5

    def __init__(self, gamma):
        """
        Parameters
        ----------
        gamma : float
            Discount factor.
        """
        self.gamma = gamma

    def evaluate(self, info, num_j_steps, is_wdr, return_Qs = False):
        """Get MAGIC estimate from Q + IPS.
        Parameters
        ----------
        info : list
            [list of actions, list of rewards, list of base propensity, list of target propensity, list of Qhat]
        num_j_steps : int
            Parameter to MAGIC algorithm
        is_wdr : bool
            Use Weighted Doubly Robust?
        return_Qs : bool
            Return trajectory-wise estimate alongside full DR estimate?
            Default: False
        
        Returns
        -------
        list
            [MAGIC estimate, normalized MAGIC, std error, normalized std error]
            If return_Qs is true, also returns trajectory-wise estimate
        """

        (actions,
        rewards,
        base_propensity,
        target_propensities,
        estimated_q_values) = MAGIC.transform_to_equal_length_trajectories(*info)

        num_trajectories = actions.shape[0]
        trajectory_length = actions.shape[1]

        j_steps = [float("inf")]

        if num_j_steps > 1:
            j_steps.append(-1)
        if num_j_steps > 2:
            interval = trajectory_length // (num_j_steps - 1)
            j_steps.extend([i * interval for i in range(1, num_j_steps - 1)])

        base_propensity_for_logged_action = np.sum(
            np.multiply(base_propensity, actions), axis=2
        )
        target_propensity_for_logged_action = np.sum(
            np.multiply(target_propensities, actions), axis=2
        )
        estimated_q_values_for_logged_action = np.sum(
            np.multiply(estimated_q_values, actions), axis=2
        )
        estimated_state_values = np.sum(
            np.multiply(target_propensities, estimated_q_values), axis=2
        )

        importance_weights = target_propensity_for_logged_action / base_propensity_for_logged_action
        importance_weights[np.isnan(importance_weights)] = 0.
        importance_weights = np.cumprod(importance_weights, axis=1)
        importance_weights = MAGIC.normalize_importance_weights(
            importance_weights, is_wdr
        )

        importance_weights_one_earlier = (
            np.ones([num_trajectories, 1]) * 1.0 / num_trajectories
        )
        importance_weights_one_earlier = np.hstack(
            [importance_weights_one_earlier, importance_weights[:, :-1]]
        )

        discounts = np.logspace(
            start=0, stop=trajectory_length - 1, num=trajectory_length, base=self.gamma
        )

        j_step_return_trajectories = []
        for j_step in j_steps:
            j_step_return_trajectories.append(
                MAGIC.calculate_step_return(
                    rewards,
                    discounts,
                    importance_weights,
                    importance_weights_one_earlier,
                    estimated_state_values,
                    estimated_q_values_for_logged_action,
                    j_step,
                )
            )
        j_step_return_trajectories = np.array(j_step_return_trajectories)

        j_step_returns = np.sum(j_step_return_trajectories, axis=1)

        if len(j_step_returns) == 1:
            weighted_doubly_robust = j_step_returns[0]
            weighted_doubly_robust_std_error = 0.0
        else:
            # break trajectories into several subsets to estimate confidence bounds
            infinite_step_returns = []
            num_subsets = int(
                min(
                    num_trajectories / 2,
                    MAGIC.NUM_SUBSETS_FOR_CB_ESTIMATES,
                )
            )
            interval = num_trajectories / num_subsets
            for i in range(num_subsets):
                trajectory_subset = np.arange(
                    int(i * interval), int((i + 1) * interval)
                )
                importance_weights = (
                    target_propensity_for_logged_action[trajectory_subset]
                    / base_propensity_for_logged_action[trajectory_subset]
                )
                importance_weights[np.isnan(importance_weights)] = 0.
                importance_weights = np.cumprod(importance_weights, axis=1)
                importance_weights = MAGIC.normalize_importance_weights(
                    importance_weights, is_wdr
                )
                importance_weights_one_earlier = (
                    np.ones([len(trajectory_subset), 1]) * 1.0 / len(trajectory_subset)
                )
                importance_weights_one_earlier = np.hstack(
                    [importance_weights_one_earlier, importance_weights[:, :-1]]
                )
                infinite_step_return = np.sum(
                    MAGIC.calculate_step_return(
                        rewards[trajectory_subset],
                        discounts,
                        importance_weights,
                        importance_weights_one_earlier,
                        estimated_state_values[trajectory_subset],
                        estimated_q_values_for_logged_action[trajectory_subset],
                        float("inf"),
                    )
                )
                infinite_step_returns.append(infinite_step_return)

            # Compute weighted_doubly_robust mean point estimate using all data
            weighted_doubly_robust, xs = self.compute_weighted_doubly_robust_point_estimate(
                j_steps,
                num_j_steps,
                j_step_returns,
                infinite_step_returns,
                j_step_return_trajectories,
            )

            # Use bootstrapping to compute weighted_doubly_robust standard error
            bootstrapped_means = []
            sample_size = int(
                MAGIC.BOOTSTRAP_SAMPLE_PCT
                * num_subsets
            )
            for _ in range(
                MAGIC.NUM_BOOTSTRAP_SAMPLES
            ):
                random_idxs = np.random.choice(num_j_steps, sample_size, replace=False)
                random_idxs.sort()
                wdr_estimate = self.compute_weighted_doubly_robust_point_estimate(
                    j_steps=[j_steps[i] for i in random_idxs],
                    num_j_steps=sample_size,
                    j_step_returns=j_step_returns[random_idxs],
                    infinite_step_returns=infinite_step_returns,
                    j_step_return_trajectories=j_step_return_trajectories[random_idxs],
                )
                bootstrapped_means.append(wdr_estimate)
            weighted_doubly_robust_std_error = np.std(bootstrapped_means)

        episode_values = np.sum(np.multiply(rewards, discounts), axis=1)
        denominator = np.nanmean(episode_values)
        if abs(denominator) < 1e-6:
            return [0]*4

        # print (weighted_doubly_robust,
        #         weighted_doubly_robust / denominator,
        #         weighted_doubly_robust_std_error,
        #         weighted_doubly_robust_std_error / denominator)

        if return_Qs:
            return [weighted_doubly_robust,
                    weighted_doubly_robust / denominator,
                    weighted_doubly_robust_std_error,
                    weighted_doubly_robust_std_error / denominator], np.dot(xs, j_step_return_trajectories)
        else:
            return [weighted_doubly_robust,
                    weighted_doubly_robust / denominator,
                    weighted_doubly_robust_std_error,
                    weighted_doubly_robust_std_error / denominator]

    def compute_weighted_doubly_robust_point_estimate(
        self,
        j_steps,
        num_j_steps,
        j_step_returns,
        infinite_step_returns,
        j_step_return_trajectories,
    ):
        low_bound, high_bound = MAGIC.confidence_bounds(
            infinite_step_returns,
            MAGIC.CONFIDENCE_INTERVAL,
        )
        # decompose error into bias + variance
        j_step_bias = np.zeros([num_j_steps])
        where_lower = np.where(j_step_returns < low_bound)[0]
        j_step_bias[where_lower] = low_bound - j_step_returns[where_lower]
        where_higher = np.where(j_step_returns > high_bound)[0]
        j_step_bias[where_higher] = j_step_returns[where_higher] - high_bound

        covariance = np.cov(j_step_return_trajectories)
        error = covariance + j_step_bias.T * j_step_bias

        # minimize mse error
        constraint = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}

        x = np.zeros([len(j_steps)])
        res = sp.optimize.minimize(
            mse_loss,
            x,
            args=error,
            constraints=constraint,
            bounds=[(0, 1) for _ in range(x.shape[0])],
        )
        x = np.array(res.x)
        return float(np.dot(x, j_step_returns)), x

    @staticmethod
    def transform_to_equal_length_trajectories(
        actions,
        rewards,
        logged_propensities,
        target_propensities,
        estimated_q_values,
    ):
        """
        Take in samples (action, rewards, propensities, etc.) and output lists
        of equal-length trajectories (episodes) accoriding to terminals.
        As the raw trajectories are of various lengths, the shorter ones are
        filled with zeros(ones) at the end.
        """
        num_actions = len(target_propensities[0][0])

        def to_equal_length(x, fill_value):
            x_equal_length = np.array(
                list(itertools.zip_longest(*x, fillvalue=fill_value))
            ).swapaxes(0, 1)
            return x_equal_length

        action_trajectories = to_equal_length(
            [np.eye(num_actions)[act] for act in actions], np.zeros([num_actions])
        )
        reward_trajectories = to_equal_length(rewards, 0)
        logged_propensity_trajectories = to_equal_length(
            logged_propensities, np.zeros([num_actions])
        )
        target_propensity_trajectories = to_equal_length(
            target_propensities, np.zeros([num_actions])
        )

        # Hack for now. Delete.
        estimated_q_values = [[np.hstack(y).tolist() for y in x] for x in estimated_q_values]

        Q_value_trajectories = to_equal_length(
            estimated_q_values, np.zeros([num_actions])
        )

        return (
            action_trajectories,
            reward_trajectories,
            logged_propensity_trajectories,
            target_propensity_trajectories,
            Q_value_trajectories,
        )

    @staticmethod
    def normalize_importance_weights(
        importance_weights, is_wdr
    ):
        if is_wdr:
            sum_importance_weights = np.sum(importance_weights, axis=0)
            where_zeros = np.where(sum_importance_weights == 0.0)[0]
            sum_importance_weights[where_zeros] = len(importance_weights)
            importance_weights[:, where_zeros] = 1.0
            importance_weights /= sum_importance_weights
            return importance_weights
        else:
            importance_weights /= importance_weights.shape[0]
            return importance_weights

    @staticmethod
    def calculate_step_return(
        rewards,
        discounts,
        importance_weights,
        importance_weights_one_earlier,
        estimated_state_values,
        estimated_q_values,
        j_step,
    ):
        trajectory_length = len(rewards[0])
        num_trajectories = len(rewards)
        j_step = int(min(j_step, trajectory_length - 1))

        weighted_discounts = np.multiply(discounts, importance_weights)
        weighted_discounts_one_earlier = np.multiply(
            discounts, importance_weights_one_earlier
        )

        importance_sampled_cumulative_reward = np.sum(
            np.multiply(weighted_discounts[:, : j_step + 1], rewards[:, : j_step + 1]),
            axis=1,
        )

        if j_step < trajectory_length - 1:
            direct_method_value = (
                weighted_discounts_one_earlier[:, j_step + 1]
                * estimated_state_values[:, j_step + 1]
            )
        else:
            direct_method_value = np.zeros([num_trajectories])

        control_variate = np.sum(
            np.multiply(
                weighted_discounts[:, : j_step + 1], estimated_q_values[:, : j_step + 1]
            )
            - np.multiply(
                weighted_discounts_one_earlier[:, : j_step + 1],
                estimated_state_values[:, : j_step + 1],
            ),
            axis=1,
        )

        j_step_return = (
            importance_sampled_cumulative_reward + direct_method_value - control_variate
        )

        return j_step_return

    @staticmethod
    def confidence_bounds(x, confidence):
        n = len(x)
        m, se = np.mean(x), sp.stats.sem(x)
        h = se * sp.stats.t._ppf((1 + confidence) / 2.0, n - 1)
        return m - h, m + h


def mse_loss(x, error):
    return np.dot(np.dot(x, error), x.T)

################################################################## Load Dataset ########################################################
############################# Define Path and Load Dataset ######################################

def load_dataset():
    path = "C:/Users/dario/Documents/DARIO/ETUDES/ICL/code/data/Round 1/"   #/mnt/c
    path2 = "C:/Users/dario/Documents/DARIO/ETUDES/ICL/code/data/Round 2/"
    pathlist = [path, path2]
    '''dic for one subject composed of ~1000 timepoints for one shot, 25 shots in one block, and 10 blocks
    First 3 blocks are baseline learning, then 6 blocks of adaptation to perturbation, and one final washout block
    That is 250 shots per subjects, 300'564 points in the dictionnary'''
    for path in pathlist:
        for i, initial in enumerate(sorted(os.listdir(path))):
            pathSubj = path + str(initial)
            for fil in range(len(sorted(os.listdir(pathSubj + '/Game/')))):
                if sorted(os.listdir(pathSubj + '/Game/'))[fil].find("Block2") > -1:
                    blockFile = sorted(os.listdir(pathSubj + '/Game/'))[fil]
                    
                    if blockFile.find('Reward') > -1:
                        print(path, initial)
                        data = resultsMultipleSubjects(path, initial, 'reward', 'all')
                        save_raw_data(data, initial)
                        save_RL_reduced_dataset(initial)
def save_raw_data(data, initial):
    reward_ = []
    cueballpos = []
    cueballvel = []
    redballpos = []
    targetcornerpos = []
    cueposfront = []
    cueposback = []
    cuedirection = []
    cuevel = []

    total_len_trajectories = 0

    for i in range(data[initial]["start_ind"].shape[0]):
        for j in range(data[initial]["start_ind"][i], data[initial]["hit_ind"][i]+1, 1):
            total_len_trajectories += 1
            if j == data[initial]["hit_ind"][i]:
                done_bool = True
                reward = data[initial]["rewards"][i]
            else:
                done_bool = False
                ## Discounted reward ##
                #gamma = 0.9
                #reward = gamma**(j - data[initial]["hit_ind"][i]) * data[initial]["rewards"][i]
                reward = 0
            reward_.append(reward)
        cueballpos.append(np.array((data[initial]["cueballpos"]["trial"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["cueballpos"]["x"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["cueballpos"]["y"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["cueballpos"]["z"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1])))
        cueballvel.append(np.array((data[initial]["cueballvel"]["trial"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["cueballvel"]["x"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["cueballvel"]["y"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["cueballvel"]["z"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1])))
        redballpos.append(np.array((data[initial]["redballpos"]["trial"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["redballpos"]["x"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["redballpos"]["y"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["redballpos"]["z"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1])))
        targetcornerpos.append(np.array((data[initial]["targetcornerpos"]["trial"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["targetcornerpos"]["x"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["targetcornerpos"]["y"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["targetcornerpos"]["z"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1])))
        cueposfront.append(np.array((data[initial]["cueposfront"]["trial"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["cueposfront"]["x"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["cueposfront"]["y"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["cueposfront"]["z"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1])))
        cueposback.append(np.array((data[initial]["cueposback"]["trial"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["cueposback"]["x"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["cueposback"]["y"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["cueposback"]["z"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1])))
        cuedirection.append(np.array((data[initial]["cuedirection"]["trial"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["cuedirection"]["x"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["cuedirection"]["y"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["cuedirection"]["z"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1])))
        cuevel.append(np.array((data[initial]["cuevel"]["trial"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["cuevel"]["x"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["cuevel"]["y"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1], data[initial]["cuevel"]["z"][data[initial]["start_ind"][i]:data[initial]["hit_ind"][i]+1])))

    dic = {'rewards': np.array(reward_), 
    'cueballpos': np.zeros(total_len_trajectories, dtype=object),
    'cueballvel': np.zeros(total_len_trajectories, dtype=object),
    'redballpos': np.zeros(total_len_trajectories, dtype=object),
    'targetcornerpos': np.zeros(total_len_trajectories, dtype=object),
    'cueposfront': np.zeros(total_len_trajectories, dtype=object),
    'cueposback': np.zeros(total_len_trajectories, dtype=object),
    'cuedirection': np.zeros(total_len_trajectories, dtype=object),
    'cuevel': np.zeros(total_len_trajectories, dtype=object)}

    transition_num = 0
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
    pd_dataset.to_csv("RL_dataset/reduced_data/"+initial+"_reduced_data.csv")
    print("reduced raw data ", initial, " saved")

def save_RL_reduced_dataset(initial):
        df = pd.read_csv("RL_dataset/reduced_data/"+initial+"_reduced_data.csv", header = 0, \
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
                'cuedirection': cuedirection
                }
        dataset = Offline_Reduced(ball_data, cue_data, cuevel, rewards[0])
        
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
        pd_dataset.to_csv("RL_dataset/Offline_reduced/"+initial+"_Offline_reduced.csv")
        print(initial, "reduced Offline dataset saved")
########################################################################################################################################
#                                                         Training
########################################################################################################################################

def train(args, **kwargs):   #config
    #np.random.seed(config.seed)
    #random.seed(config.seed)
    #torch.manual_seed(config.seed)      

    ############################# Define Path and Load Dataset ######################################

	path = "/mnt/c/Users/dario/Documents/DARIO/ETUDES/ICL/code/data/Round 1/"
	path2 = "/mnt/c/Users/dario/Documents/DARIO/ETUDES/ICL/code/data/Round 2/"
    ## WARNING check which type of feedback in which round, and which corner for each subject
    #corner =  "left"
    #dic = resultsSingleSubject("BL", "reward", path, corner)
	'''dic for one subject composed of ~1000 timepoints for one shot, 25 shots in one block, and 10 blocks
    First 3 blocks are baseline learning, then 6 blocks of adaptation to perturbation, and one final washout block
    That is 250 shots per subjects, 300'564 points in the dictionnary'''

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
		"discount": args.discount,
		"tau": args.tau,
		# TD3
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		# TD3 + BC
		"alpha": args.alpha
	    }
    
	# Initialize Agent
	policy = TD3_BC(**kwargs)
	'''
	if args.policy == "TD3_BC":
		policy = TD3_BC(**kwargs) 
	elif args.policy == "CQL_SAC":
		policy = CQLSAC(state_size=state_dim, action_size=action_dim, device=device)
	else:
		raise ValueError("Chose Agent between [TD3_BC, CQL_SAC]")
    
	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")
    '''

    # Dataframe for all subjects
	#data = resultsMultipleSubjects([path, path2], 'reward', 'all')

	#dataset = Offline_RL_dataset(data, terminate_on_end=True)

    ############### Load Saved dataset ####################
    # Read Saved dataset
	df = pd.read_csv("RL_dataset/AAB.csv", header = 0, \
            names = ['trial','states','actions','new_states','rewards','terminals'], usecols = [1,2,3,4,5,6], lineterminator = "\n")
	df = df.replace([r'\n', r'\[', r'\]'], '', regex=True) 
	observations = pd.DataFrame.from_records(np.array(df['states'].str.split(','))).astype(float)
	actions= pd.DataFrame.from_records(np.array(df['actions'].str.split(','))).astype(float)
	next_observations = pd.DataFrame.from_records(np.array(df['new_states'].str.split(','))).astype(float)

	dic = {'trial': df["trial"].to_numpy(),
        'observations': observations.to_numpy(),
		'actions': actions.to_numpy(),
		'next_observations': next_observations.to_numpy(),
		'rewards': df["rewards"].to_numpy(),
		'terminals': df["terminals"].to_numpy()}

	replay_buffer = ReplayBuffer(state_dim, action_dim)
	replay_buffer.convert_D4RL(dic)	#dataset)

	if normalize:
		mean,std = replay_buffer.normalize_states() 
	else:
		mean,std = 0,1
	
	################## Training #######################
	#for t in range(int(args.max_timesteps)):
		#policy.train(replay_buffer, args.batch_size)


	steps = 0
	average10 = deque(maxlen=10)
	total_steps = 0
	batch_size = 64 #256

	for i in range(1, 100):
		episode_steps = 0
		rewards = 0
		while True:
			steps += 1
			
			#state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
			#policy_loss, alpha_loss, bellmann_error1, bellmann_error2, cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha = policy.train(steps, state, action, reward, next_state, not_done, gamma=0.99)
			policy.train(replay_buffer)
			#state = next_state
			#rewards += reward
			episode_steps += 1
			#if not_done == False:
				#break


		average10.append(rewards)
		total_steps += episode_steps
		print("Episode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, reward, policy_loss, steps))
    
########################################################################################################################################
#                                                         Main
########################################################################################################################################

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="TD3_BC") #CQL_SAC            # Policy name
	parser.add_argument("--env", default="hopper-medium-v0")        # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	# TD3
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	# TD3 + BC
	parser.add_argument("--alpha", default=2.5)
	parser.add_argument("--normalize", default=True)
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	#train(args)
	load_dataset()
	################## Validation #######################
	""""
	evaluations = []
	# Evaluate episode
	for t in range(int(args.eval_max_timesteps)):
		print(f"Time steps: {t+1}")
		# TO DO: Define an Evaluation Policy
		evaluations.append(eval_policy(policy, args.env, args.seed, mean, std))
		np.save(f"./results/{file_name}", evaluations)
		if args.save_model: policy.save(f"./models/{file_name}")"""
