from model import *
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def extract(df,newDf):

    for i in df.index:
        states = np.array(df["state"].loc[i])
        actions = np.array(df["action"].loc[i]).reshape(len(df["action"].loc[i]),1)
        iteration = np.concatenate((states,actions),axis=1)
        new_states = np.array(df["next_state"].loc[i])
        iteration = np.concatenate((iteration,new_states),axis=1)
        time = np.arange(len(states)).reshape(len(states),1)
        iteration = np.concatenate((iteration,time),axis=1)
        newDf = pd.concat((newDf,pd.DataFrame(iteration)),axis=0)
    newDf=newDf.reset_index(drop=True)
    return newDf

def customReward(state):
    maxX = 2.4
    minX = -2.4
    maxDX = 10
    minDX = -10
    maxTheta = 0.2095
    minTheta = -0.2095
    maxDTheta = 10
    minDTheta  = -10
    goalX = abs(state[0])
    goalDX = abs(state[1])
    goalTheta = abs(state[2])
    goalDTheta = abs(state[3])
    return goalX+goalDX+goalTheta+goalDTheta

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

env = gym.make("CartPole-v1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','count','episode'))

n_nodes = 64
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
simEpisodes = 20


# Get number of actions from gym action space
n_actions = env.action_space.n

# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

transferFunctionNetwork = XTFC_S(n_nodes=n_nodes,input_size=n_actions+1,output_size=n_observations,length=0)
qValueNetwork = XTFC_Q(n_nodes=n_nodes,input_size=n_observations,output_size=n_actions,length=0,epsilon=1)

memory = ReplayMemory(10000)
episode_durations = []
data = {}
for i in range(simEpisodes):
    count = 0
    done = False
    state,info = env.reset()
    data[i]={"state":[],"action":[],"next_state":[]}
    while not done:
        if qValueNetwork.epsilon<random.random():
            pred =qValueNetwork.pred(x=state).detach()
            actionPrediction = np.argmax(pred[:,0].numpy())
        else: 
            actionPrediction = random.choice([0,1])
            qValueNetwork.epsilon += qValueNetwork.epsilon -qValueNetwork.epsilon_decay
            
        
        next_state, reward, terminated, truncated, _ = env.step(actionPrediction)
        memory.push(state, actionPrediction, next_state, reward,count,i)
        data[i]["state"].append(state)
        data[i]["action"].append(actionPrediction)
        data[i]["next_state"].append(next_state)
        
        state = next_state
        done = terminated or truncated

        if terminated:
            next_state = None

        if done:
            episode_durations.append(count + 1)
            done=True
        count+= 1

df = pd.DataFrame.from_dict(data).T
new_df = pd.DataFrame()
new_df = extract(df,new_df)
print(new_df[4])
# transferFunctionNetwork.train(100,5,)
    



