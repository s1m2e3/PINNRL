from model import *
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
                        ('state', 'action', 'next_state', 'reward'))

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
n_observations = int(len(state)/2)

transferFunctionNetwork = XTFC_S(n_nodes=n_nodes,input_size=n_actions+1,output_size=n_observations,length=0)
qValueNetwork = XTFC_Q(n_nodes=n_nodes,input_size=n_actions+1,output_size=n_observations,length=0)

memory = ReplayMemory(10000)

for i in range(simEpisodes):
    actionPrediction = np.argmax(XTFC_Q.pred(state)[:,0].numpy())
    next_state, reward, terminated, truncated, _ = env.step(actionPrediction)
    memory.push(state, actionPrediction, next_state, reward)
    state = next_state
    if terminated:
        

    



