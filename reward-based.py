import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt

import seaborn as sns
import statsmodels.api as sm

import scipy.stats as ss
import pingouin as pg

import scipy.signal as sci
import math
import scipy as sp

import itertools

from datetime import datetime, timedelta
from datetime import date

import re

#import d4rl
import copy
import argparse
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################################################################################################################################
#                                                         Functions
########################################################################################################################################


#################################### Import Data Functions ############################################

# Error task

def importDataError (initials, path, pocket, lbThreshold = -9.364594, ubThreshold = 4.675092):

    '''Imports Behavioural Data for a subject performing Error task

    Args:
    - initials (str): initials of the subject
    - path (str): path of the data
    - pocket (str): left/right corresponding to the pocket
    - lbThreshold (float), ubThreshold (float): 
        lower and upper bounds of the interval of possible angles (out of the interval it will be considered outlier)
    
    Output:
    - errorSubjectData (dict): combinations initials (key) - dataframe (value) for each subject with chosen pocket for error task.
        Every entry of the dictionary stores three data frames:
            1. VRData: behavioural data from the unity game
            2. angleData: angle of the shot and corresponding trial numbers (250x2)
            3. successData: binary variable for success and corresponding trial numbers (250x2)
    
    '''
    
    print("Import Data function")
    # Declare possible pocket and raise error otherwise
    pockets = ['left', 'right']
    if pocket.lower() in pockets:
        pocket = pocket.capitalize()
    else:
        raise ValueError("Invalid game type. Expected one of: %s" % pockets)


    ### Definition of ideal angle and funnel of pocketing ###
    idealAngle = 105.0736
    lb_idealAngle = 104.5401 - idealAngle
    ub_idealAngle = 105.6072 - idealAngle


    ### Import angle data ###
    angleData = pd.read_csv(path + initials + "/Game/" + initials + "_Angle.txt", header=None, names = ['Angle'])
    angleData['Block'] = np.repeat(range(1,11),25)
    # Remove outliers setting value to nan
    angleData.loc[(angleData.Angle - idealAngle) > ubThreshold, 'Angle'] = np.nan
    angleData.loc[(angleData.Angle - idealAngle) < lbThreshold, 'Angle'] = np.nan   


    ### Import success data ###
    successData = pd.read_csv(path + initials + "/Game/" + initials + "_Success.txt",  sep = '\t', header = None, \
        names = ['Block','Trial','Angle','Magnitude','RBPosition'], usecols = [0,1,2,3,5])



    ### Import Behavioural Data from Blocks ###
    VRData = pd.DataFrame()

    # Merge all blocks together
    for bl in range(1,11):
        block = pd.read_csv(path + initials + "/Game/" + initials + "_Error" +  pocket + "_Block" + str(bl) + ".txt", sep = '\t')
        # remove shots after 25 trials (done by mistake)
        while block['TrialNumber'].iloc[-1]!=25:
            block.drop(block.tail(1).index,inplace=True)

        block['TrialNumber'] = block['TrialNumber'] + (bl-1)*25  
        VRData = pd.concat([VRData, block])


    ### Store data in a dictionary ###
    errorSubjectData = {}
    names = ['VRData','Angle','Success']
    dfs = [VRData, angleData, successData]
    counter = 0

    for df in names:
        errorSubjectData[df] = dfs[counter]
        counter += 1

    ### Return the dictionary ###
    return(errorSubjectData)


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

def preprocess_old(dataset):

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
    dataset['redballpos_str'] = dataset['redballpos'].str.split(',')
    dataset['stick'] = dataset['optifront'].str.split(',')
    dataset['gaze_str'] = dataset['gaze'].str.split(',')


    ### Create datasets from string variables ###
    cbpos = pd.DataFrame.from_records(np.array(dataset['cueballpos_str']), columns=['x','y','z']).astype(float)
    rbpos = pd.DataFrame.from_records(np.array(dataset['redballpos_str']), columns=['x','y','z']).astype(float)
    stick = pd.DataFrame.from_records(np.array(dataset['stick']), columns=['x','y','z']).astype(float)
    gaze = pd.DataFrame.from_records(np.array(dataset['gaze_str']), columns=['x','y','z']).astype(float)


    ### Standardise w.r.t cue ball initial position and add time and trial number ###
    x_std, y_std, z_std = cbpos.iloc[0]

    for df in (cbpos, rbpos, stick, gaze):
        df -= (x_std, y_std, z_std)

        df['trial'] = np.array(dataset['TrialNumber'])
        df['time'] = np.array(dataset['TrialTime']) 
        df['timeReal'] = np.array(dataset['GlobalTime'])   


    ### Create a dictionary for saving dataframes and save them ###
    subject = {}
    names = ['cbpos','rbpos','stick','gaze']
    dfs = [cbpos, rbpos, stick, gaze]
    counter = 0

    for df in names:
        subject[df] = dfs[counter]
        counter += 1


    ### Return the dictionary ###
    return(subject)


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

#################################### Succes Derivation ##########################################

def successTrialsError(subject):

    '''Preprocesses and derive successful trials from VR Data correcting the output for error subjects
    
    Args:
    - subject (dict): output of importDataError or importDataReward
    
    Output:
    - subject (dict): copy of the input with correct success definition
    
    '''

    idealAngle = 105.0736
    lb_idealAngle = 104.5401 - idealAngle
    ub_idealAngle = 105.6072 - idealAngle

    
    ### Derive Successful trials from Angle dataset
    subject['Angle']['Success'] = subject['Angle'].Angle.isin(subject['Success']['Angle']).astype(int)
    subject['Angle']['Trial'] = range(1,251)
    subject['Angle']['AngleStd'] =  subject['Angle']['Angle'] - idealAngle

    gameData = preprocess(subject['VRData'])

    succTrials = subject['Angle'].Success.index[subject['Angle'].Success == 1] + 1

    for tr in succTrials:
        cbTrial = gameData['cbpos'][gameData['cbpos'].trial == tr]

        if (cbTrial['y'].iloc[0] - cbTrial['y'].iloc[-1]) >= 0.02:
            subject['Angle'].Success.iloc[tr-1] = 0
    print(str(subject['Success'].shape[0]-sum(subject['Angle'].Success)) + " fake successes removed")


    # Correction if missing successes in perturbation phase for error
    df = subject['Angle'][subject['Angle']['Block'] > 3]
    df = df[df['Block'] < 10]
    if df['Success'].sum()==0:
      for tr in range(76, 226):
          rbTrial = gameData['rbpos'][gameData['rbpos'].trial == tr]
          if (np.abs(np.median(rbTrial['y'].tail(10)) - np.median(rbTrial['y'].head(rbTrial.shape[0]-10))) > 0.02) and (np.abs(np.median(rbTrial['y'].tail(10)) - np.median(rbTrial['y'].head(rbTrial.shape[0]-10))) < 0.5):
              subject['Angle'].Success.iloc[tr-1] = 1
    
    ### Returns preprocessed data with correct success
    return(subject)


def successTrialsReward(subject, preprocessData):

  '''Preprocesses and derive successful trials from VR Data correcting the output for reward subjects, \
    creating three new variables in the Angle dataset: SuccessMedian (binary variable for success attributable to the median), \
    SuccessFunnel (binary variable for success attributable to the funnel) and Target (reference angle to be rewarded)
  
    Args:
    - subject (dict): output of importDataError or importDataReward
    
    Output:
    - subject (dict): copy of the input with correct success definition and three new variables in the Angle dataset
   
  '''
  
  print("Success Derivation function")

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
    subject['Angle']['Success'].iloc[tr] = 1

    if (subject['Motivation'].Motivation[subject['Motivation'].Trial == tr] == 'Median').all():
      subject['Angle']['SuccessMedian'][tr] = 1
    else:
      subject['Angle']['SuccessFunnel'][tr] = 1 


  ### Derive target for reward (median of the past 10 successful trials) ###

  target = []
  vec_median = list(subject['Angle']['AngleStd'][range(66,76)])

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

############################ Define Dataframe for all subjects ##################################

def resultsMultipleSubjectsWriteDF_old(pathList, gameType, pocketSide):

    '''
    Args:
    - pathList (list): list of all paths of the folder with raw data
    - gameType (str): error/reward - game type 
    - pocketSide (str): left/right - game pocket
 
    '''
	
    # Declare possible gameType and raise error otherwise
    game_types = ['error', 'reward']
    if gameType.lower() in game_types:
        mode = gameType.lower()
    else:
        raise ValueError("Invalid game type. Expected one of: %s" % game_types)

    # Call dataframe for both rounds together
    cueballPos = pd.DataFrame()
    start_ind = pd.DataFrame()
    hit_ind = pd.DataFrame()
    redballPos = pd.DataFrame()
    targetcornerPos = pd.DataFrame()
    cuePosfront = pd.DataFrame()
    cuePosback = pd.DataFrame()
    cueDirection = pd.DataFrame()
    cueVel = pd.DataFrame()
    if gameType == "error":
        errors = pd.DataFrame()
    else:
        rewards = pd.DataFrame()
    
    
    # Declare possible pocketSide and raise error otherwise
    pocketChoice = ['left', 'right', 'all']
    if pocketSide.lower() in pocketChoice:
        pocket = pocketSide.lower()
    else:
        raise ValueError("Invalid pocket. Expected one of: %s" % pocketChoice)


    #subjError = {}
    subjReward = {}


    for path in pathList:

        for sub in next(os.walk(path))[1]:
            """
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

                        # Add data to dataframes previously defined
                        if round == 'first':
                            anglesErrorR1[str(initials)] = subjData['Game']['Angle']['AngleStd']
                            successErrorR1[str(initials)] = subjData['Game']['Angle']['Success']
                        elif round == 'second':
                            anglesErrorR2[str(initials)] = subjData['Game']['Angle']['AngleStd']
                            successErrorR2[str(initials)] = subjData['Game']['Angle']['Success']
                        elif round == 'both':
                            anglesError[str(initials)] = subjData['Game']['Angle']['AngleStd']
                            successError[str(initials)] = subjData['Game']['Angle']['Success']


                        print("Imported " + initials + " as error subject " + "for " + pocketSub + " pocket")
                """
        

        
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

                        # Add data to dataframes previously defined
                        sentence = subjData['Game']["VRData"]["cueballpos"]
                        cueballPos[str(initials)] = [float(s) for s in re.findall(r'-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?', sentence)]
                        
                        start_ind[str(initials)], hit_ind[str(initials)] = start_hit_timesteps(cueballPos[str(initials)], subjData["Game"]["VRData"]["TrialNumber"])
                        
                        sentence = subjData['Game']["VRData"]["redballpos"]
                        redballPos[str(initials)] = [float(s) for s in re.findall(r'-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?', sentence)]

                        if pocketSub == 'Left':
                            sentence = subjData['Game']["VRData"]["corner5pos"]
                        else:
                            sentence = subjData['Game']["VRData"]["corner6pos"]
                        targetcornerPos[str(initials)] = [float(s) for s in re.findall(r'-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?', sentence)] 

                        sentence = subjData['Game']["VRData"]["cueposfront"]
                        cuePosfront[str(initials)] = [float(s) for s in re.findall(r'-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?', sentence)]

                        sentence = subjData['Game']["VRData"]["cueposback"]
                        cuePosback[str(initials)] = [float(s) for s in re.findall(r'-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?', sentence)]

                        sentence = subjData['Game']["VRData"]["cuedirection"]
                        cueDirection = [float(s) for s in re.findall(r'-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?', sentence)]

                        sentence = subjData['Game']["VRData"]["cuevel"]
                        cueVel[str(initials)] = [float(s) for s in re.findall(r'-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?', sentence)]

                        rewards[str(initials)] = def_reward(subjData['Game']["Angle"]["Success"], subjData["Game"]["Angle"]["SuccessFunnel"], subjData["Game"]["Angle"]["SuccessMedian"])

                        subjReward[initials] = subjData

                        print("Imported " + initials + " as reward subject " + "for " + pocketSub + " pocket" + " in round: " + path)
            
            start_ind, hit_ind = start_hit_timesteps(cueballPos, subjData["Game"]["VRData"]["TrialNumber"])
    return cueballPos, redballPos, targetcornerPos, cuePosfront, cuePosback, cueDirection, cueVel, rewards, start_ind, hit_ind



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


def resultsMultipleSubjects(pathList, gameType, pocketSide):
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


    k=0
    for path in pathList:
        for sub in next(os.walk(path))[1]:
            print(sub)
            if k < 1:
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

                            # Add data to dataframes previously defined
                            dataset[str(initials)]={}

                            dataset[str(initials)]["rewards"] = def_reward(subjData["Game"]["Angle"]["SuccessMedian"], subjData["Game"]["Angle"]["SuccessFunnel"])
             

                            dataset[str(initials)]["cueballpos"] = subjData["PreProcessed"]["cbpos"]
                            
                            dataset[str(initials)]["start_ind"], dataset[str(initials)]["hit_ind"] = start_hit_timesteps(subjData["PreProcessed"])   #, subjData["Game"]["VRData"]["TrialNumber"])
                        
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
                            k+=1
            else:
                break
            
    return dataset

############################# Find Start and hitting cueball timesteps #############################
'''def variability_cue_movement(subjData):
    cueballPos = subjData["cbpos"]
    cueballvel = subjData["cbvel"]
    redballvel = subjData["rbvel"]
    cuevel = subjData["cuevel"]
    for i in range(len(cueballPos)):
        if i != 0:
            #New trial started
            if cueballPos["trial"].iloc[i] > prev_trial:
                #shooting along z-axis so low variability along x and y-axis   
                if np.linalg.norm(cueballvel[["x", "y", "z"]].iloc[i].to_numpy()) < variability and np.linalg.norm(cueballvel[["x", "y", "z"]].iloc[i-1].to_numpy()) < err and np.linalg.norm(redballvel[["x", "y", "z"]].iloc[i].to_numpy()) < err and np.linalg.norm(redballvel[["x", "y", "z"]].iloc[i-1].to_numpy()) < err and np.linalg.norm(cuevel[["x", "y", "z"]].iloc[i].to_numpy()) < err:    #and cueballvel["y"].iloc[i-1] == 0  and redballvel["y"].iloc[i-1] == 0
                    print("a")
            elif cueballPos["trial"].iloc[i] == prev_trial:
                if np.linalg.norm(cueballvel[["x", "y", "z"]].iloc[i].to_numpy()) != 0 and block_until_next_trial == False:
                    print("a")'''

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

def start_hit_timesteps(subjData): #, TrialNumber):
    
    print("start hit timesteps function")
    cueballPos = subjData["cbpos"]
    cueballvel = subjData["cbvel"]
    redballvel = subjData["rbvel"]
    cuevel = subjData["cuevel"]
    #Find when cue hits cueball
    #trial=0
    hit_ind=[]
    start_ind=[]
    n = 1
    threshold = 0.1

    #hit_ind=np.zeros(250)
    #start_ind=np.zeros(250)
    #Bool for missed hit
    miss_hit = False

    #pos = cueballPos.iloc[0]

    prev_trial = 0
    block_until_next_trial = True
    #k=0
    for i in range(len(cueballPos)):
        #The first 200 timesteps approximately are calibration and parasite movements
        if i > 200:   #!= 0:
            #New trial started
            if cueballPos["trial"].iloc[i] > prev_trial:
                if miss_hit == True and len(hit_ind) < len(start_ind):   
                    hit_ind = np.append(hit_ind, start_ind[-1]+350)#on average hitting cueball after 350 timesteps
                    #hit_ind[k] = start_ind[-1]+350
                    #k+=1
                    miss_hit == False 
                if static_for_n_timesteps(cueballvel, redballvel, i, n):
                    #Wait for cueball vel y-axis and redball vel y-axis to be zero after new trial started
                    start_ind = np.append(start_ind, i)
                    #start_ind[k] = i
                    prev_trial = cueballPos["trial"].iloc[i]
                    block_until_next_trial = False
                    miss_hit = True
            elif cueballPos["trial"].iloc[i] == prev_trial:
                if np.linalg.norm(cueballvel[["x", "y", "z"]].iloc[i].to_numpy()) > threshold and block_until_next_trial == False:
                #(cueballPos["x"].iloc[i-1] != cueballPos["x"].iloc[i] or cueballPos["z"].iloc[i-1] != cueballPos["z"].iloc[i]) and cueballPos["trial"].iloc[i] == prev_trial and block_until_next_trial == False:
                    #hit_ind = np.append(hit_ind, i+5)#Add 6 timesteps for margin
                    hit_ind = np.append(hit_ind, i+5)
                    #hit_ind[k] = i+5
                    block_until_next_trial = True
                    miss_hit = False
                    #k+=1
        #if last Trial is missed
        if i == len(cueballPos.index) and miss_hit:
            hit_ind = np.append(hit_ind, start_ind[-1]+350)#on average hitting cueball after 350 timesteps
            #hit_ind[k] = start_ind[-1]+350

    if len(start_ind) != 250 or len(hit_ind) != 250:
        raise ValueError( "Missed an index. start ind size", len(start_ind), "and hit ind size: ", len(hit_ind))
    for i in range(len(start_ind)):
        if start_ind[i] >= hit_ind[i]:
            raise ValueError("start ind > hit_ind", i , start_ind, hit_ind)
    return start_ind.astype(int), hit_ind.astype(int)

#################################### Define Actions ########################################

def compute_impulseForce(cuevel, cuedirection):
    #numVelocitiesAverage = 5
    
    impulseForce = np.zeros(cuevel.shape)
    shotMagnitude = np.zeros(1)
    shotDir = np.zeros(cuedirection.shape)
    #Reward: magnitude range
    lbMagnitude = 0.516149
    ubMagnitude = 0.882607

    shotMagnitude = np.linalg.norm(cuevel)
    if shotMagnitude > ubMagnitude:
        shotMagnitude = ubMagnitude
        #print("upper bounded")
    elif shotMagnitude < lbMagnitude:
        shotMagnitude = lbMagnitude
        #print("lower bounded")

    for i in range(len(cuevel)):
        '''if i < numVelocitiesAverage:
            VelocityList = cuevel[["x","z"]].iloc[:i]   #Along y axis as well?
            cueDirList = cuedirection[["x","z"]].iloc[:i]
        else:
            VelocityList = cuevel[["x","z"]].iloc[i-(numVelocitiesAverage+1):i+1]
            cueDirList = cuedirection[["x","z"]].iloc[i-(numVelocitiesAverage+1):i+1]'''
    
    #avgVelocity = np.median(VelocityList) #median cue stick velocity from last 10 frames (~0.11 sec)
    #avgDir = np.median(cueDirList) #median direction of cue stick from last 10 frames

        shotDir[i] = cuedirection.iloc[i]
        impulseForce[i] = shotMagnitude * shotDir[i]
    return impulseForce

#################################### Define Reward #########################################

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

############################## Build RL Offline Dataset #########################################

def Offline_RL_dataset(data, terminate_on_end=False):
    """
        Returns datasets formatted for use by standard Q-learning algorithms,
        with observations, actions, next_observations, rewards, and a terminal
        flag.

        Args:
            cueballPos, redballPos, targetcornerPos, cuePosfront, cuePosback: Recorded Observation States (for each subject)
            cueVel: Recorded Action State (for each Subject)
            reward: np.array with one reward per trial
            start_ind: starting index of each trial
            hit_ind: hitting the cue ball index for each trial
            target_corner: string indicating the corner in which to pocket
            terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.

        Returns:
            A dictionary containing keys:
                observations: An N x dim_obs array of observations.
                actions: An N x dim_action array of actions.
                next_observations: An N x dim_obs array of next observations.
                rewards: An N-dim float array of rewards.
                terminals: An N-dim boolean array of "done" or episode termination flags.
        """
    
    print("Offline RL Dataset function")
    N = len(data["AAB"]["start_ind"])
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    obs = np.zeros(14)  #np.zeros(21)
    new_obs = np.zeros(14)  #np.zeros(21)
    action = np.zeros(2)
    reward = np.zeros(1)
    gamma = 0.6 #discounted reward
    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    #use_timeouts = False
    #if 'timeouts' in dataset:
        #use_timeouts = True
    
    '''terminate_on_end (bool): Set done=True on the last timestep
        in a trajectory. Default is False, and will discard the
        last timestep in each trajectory.'''

    episode_step = 0
    for i in range(N-1):
        for j in range(data["AAB"]["start_ind"][i], data["AAB"]["hit_ind"][i], 1):
            #Observation timestep t
            obs[0] = data["AAB"]["cueballpos"]["x"].iloc[j] 
            #obs[0] = data["AAB"]["cueballpos"][y].iloc[j] 
            obs[1] = data["AAB"]["cueballpos"]["z"].iloc[j] 
            obs[2] = data["AAB"]["redballpos"]["x"].iloc[j] 
            #obs[2] = data["AAB"]["redballpos"][y].iloc[j]
            obs[3] = data["AAB"]["redballpos"]["z"].iloc[j]
            obs[4] = data["AAB"]["targetcornerpos"]["x"].iloc[j]
            obs[4] = data["AAB"]["targetcornerpos"]["y"].iloc[j]
            obs[5] = data["AAB"]["targetcornerpos"]["z"].iloc[j]
            obs[6] = data["AAB"]["cueposfront"]["x"].iloc[j]
            #obs[6] = data["AAB"]["cueposfront"][y].iloc[j]
            obs[7] = data["AAB"]["cueposfront"]["z"].iloc[j]
            obs[8] = data["AAB"]["cueposback"]["x"].iloc[j]
            #obs[8] = data["AAB"]["cueposback"][y].iloc[j]
            obs[9] = data["AAB"]["cueposback"]["z"].iloc[j]
            obs[10] = data["AAB"]["cuedirection"]["x"].iloc[j]
            #obs[10] = data["AAB"]["cueDirection"][y].iloc[j]
            obs[11] = data["AAB"]["cuedirection"]["z"].iloc[j]
            obs[12] = data["AAB"]["cuevel"]["x"].iloc[j]
            obs[13] = data["AAB"]["cuevel"]["z"].iloc[j]

            #Observations Timestep t+1
            new_obs[0] = data["AAB"]["cueballpos"]["x"].iloc[j+1] 
            #new_obs[0] = data["AAB"]["cueballpos"][y].iloc[j+1] 
            new_obs[1] = data["AAB"]["cueballpos"]["z"].iloc[j+1] 
            new_obs[2] = data["AAB"]["redballpos"]["x"].iloc[j+1] 
            #new_obs[2] = data["AAB"]["redballpos"][y].iloc[j+1]
            new_obs[3] = data["AAB"]["redballpos"]["z"].iloc[j+1]
            new_obs[4] = data["AAB"]["targetcornerpos"]["x"].iloc[j+1]
            new_obs[4] = data["AAB"]["targetcornerpos"]["y"].iloc[j+1]
            new_obs[5] = data["AAB"]["targetcornerpos"]["z"].iloc[j+1]
            new_obs[6] = data["AAB"]["cueposfront"]["x"].iloc[j+1]
            #new_obs[6] = data["AAB"]["cueposfront"][y].iloc[j+1]
            new_obs[7] = data["AAB"]["cueposfront"]["z"].iloc[j+1]
            new_obs[8] = data["AAB"]["cueposback"]["x"].iloc[j+1]
            #new_obs[8] = data["AAB"]["cueposback"][y].iloc[j+1]
            new_obs[9] = data["AAB"]["cueposback"]["z"].iloc[j+1]
            new_obs[10] = data["AAB"]["cuedirection"]["x"].iloc[j+1]
            #new_obs[10] = data["AAB"]["cueDirection"][y].iloc[j+1]
            new_obs[11] = data["AAB"]["cuedirection"]["z"].iloc[j+1]
            new_obs[12] = data["AAB"]["cuevel"]["x"].iloc[j+1]
            new_obs[13] = data["AAB"]["cuevel"]["z"].iloc[j+1]

            #Action Velocity, Force?
            action = compute_impulseForce(data["AAB"]["cuevel"][["x", "z"]].iloc[j], data["AAB"]["cuedirection"][["x", "z"]].iloc[j])
            #action[0] = data["AAB"]["cuevel"]["y"].iloc[j]
            
            if j== data["AAB"]["hit_ind"][i]:
                done_bool = True
                reward = data["AAB"]["rewards"][i]
            else:
                done_bool = False
                reward = 0

            final_timestep = (episode_step == data["AAB"]["hit_ind"][i]-1)
            if (not terminate_on_end) and final_timestep:
                # Skip this transition and don't apply terminals on the last step of an episode
                episode_step = 0
                continue
            if done_bool or final_timestep:
                episode_step = 0

            obs_.append(obs)
            next_obs_.append(new_obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done_bool)
            episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        }

################################### Replay Buffer ##############################################

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def convert_D4RL(self, dataset):
		self.state = dataset['observations']
		self.action = dataset['actions']
		self.next_state = dataset['next_observations']
		self.reward = dataset['rewards'].reshape(-1,1)
		self.not_done = 1. - dataset['terminals'].reshape(-1,1)
		self.size = self.state.shape[0]


	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.state = (self.state - mean)/std
		self.next_state = (self.next_state - mean)/std
		return mean, std

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


################################### CQL_SAC AGENT ##############################################
'''
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
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

    def _compute_policy_values(self, obs_pi, obs_q):
        with torch.no_grad():
            actions_pred, log_pis = self.actor_local.evaluate(obs_pi)
        
        qs1 = self.critic1.get_qvalues(obs_q, actions_pred)
        qs2 = self.critic2.get_qvalues(obs_q, actions_pred)
        
        return qs1-log_pis, qs2-log_pis
    
    def _compute_random_values(self, obs, actions, critic):
        random_values = critic.get_qvalues(obs, actions)
        random_log_probs = math.log(0.5 ** self.action_size)
        return random_values - random_log_probs
    
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

	train(args)
	
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
