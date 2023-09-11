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

def predictFuture(qNetwork,transferNetwork,state):
    actions = []
    time = np.arange(50)
    for i in range(50):
        pred = qValueNetwork.pred(x=state).detach()
        actionPrediction = np.argmax(pred[:,0].numpy())
        # new_states = transferNetwork.pred
    return qNetwork


def trainQExistingData(network,new_df):
    for j in new_df[11].unique():
        
        sub_df = new_df[new_df[11]==j]
        current_state = np.array(sub_df[[0,1,2,3]])
        next_state = np.array(sub_df[[5,6,7,8]])
        rewards = np.array(sub_df[9])
        q_pred = network.pred(x=next_state).detach().numpy()
        q_indexes = np.argmax(q_pred,axis=1)
        y_train = np.array([q_pred[i,q_indexes[i]]for i in range(len(q_pred))])*GAMMA+rewards
        _,_,_,_=qValueNetwork.train(accuracy=accuracy,n_iterations=iterations,x_train=current_state,y_train=y_train)
    return network

def extract(df,newDf):

    for i in df.index:
        states = np.array(df["state"].loc[i])
        actions = np.array(df["action"].loc[i]).reshape(len(df["action"].loc[i]),1)
        iteration = np.concatenate((states,actions),axis=1)
        new_states = np.array(df["next_state"].loc[i])
        iteration = np.concatenate((iteration,new_states),axis=1)
        rewards = np.array(df["reward"].loc[i]).reshape(len(df["reward"].loc[i]),1)
        iteration = np.concatenate((iteration,rewards),axis=1)
        time = np.arange(len(states)).reshape(len(states),1)
        iteration = np.concatenate((iteration,time),axis=1)
        episode = np.ones(len(states)).reshape(len(states),1)*i
        iteration = np.concatenate((iteration,episode),axis=1)
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

n_nodes = 512
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
simEpisodes = 1000
simTrainings = 100

# Get number of actions from gym action space
n_actions = env.action_space.n

# Get the number of state observations
state, info = env.reset()
n_observations = len(state)
qValueNetwork = XTFC_Q(n_nodes=n_nodes,input_size=n_observations,output_size=n_actions,length=0,epsilon=1)
transferFunctionNetwork = XTFC_S(n_nodes=64,input_size=n_actions,output_size=int(n_observations/2),length=0)
memory = ReplayMemory(10000)
episode_durations = []
data = {}
total_rewards={"elm":{},"random":{}}
plt.figure()
for name in ["elm",'random']:
    
    for i in range(simEpisodes):
        count = 0
        done = False
        state,info = env.reset()
        data[i]={"state":[],"action":[],"next_state":[],"reward":[]}
        # if i % 21==0 and i>0:
            # qValueNetwork=predictFuture(transferFunctionNetwork,qValueNetwork,state)
    
        
        while not done:
            if qValueNetwork.epsilon<random.random():
                pred = qValueNetwork.pred(x=state).detach()
                actionPrediction = np.argmax(pred[:,0].numpy())
                # input("hey")
            
            else: 
            
                actionPrediction = random.choice([0,1])
                qValueNetwork.epsilon -= qValueNetwork.epsilon_decay

            next_state, reward, terminated, truncated, _ = env.step(actionPrediction)
            memory.push(state, actionPrediction, next_state, reward,count,i)
            data[i]["state"].append(state)
            data[i]["action"].append(actionPrediction)
            data[i]["next_state"].append(next_state)
            data[i]["reward"].append(reward)
            
            state = next_state
            done = terminated or truncated

            if terminated:
                next_state = None

            if done:
                episode_durations.append(count + 1)
                done=True
            count+= 1

        total_rewards[name][i] = sum(data[i]["reward"])
        
        if  i%80==0 and i>0:
            episodes = list(data)
            passed_data = episodes[-80:]
            res = dict((k, data[k]) for k in passed_data if k in data)
            df = pd.DataFrame.from_dict(res).T
            accuracy = 1e-4
            iterations = 1
            new_df = pd.DataFrame()
            new_df = extract(df,new_df)
            
            for i in new_df[11].unique():
                sub_df = new_df[new_df[11]==i]
                x_train = np.array(sub_df[[9,4]])
                y_train_t = np.array(sub_df[[5,7]])
                y_train_dt = np.array(sub_df[[6,8]])
                y_train = np.concatenate((y_train_t,y_train_dt))
                transferFunctionNetwork.train(accuracy=accuracy,n_iterations=iterations,x_train=x_train,y_train=y_train)
            
            qValueNetwork=trainQExistingData(qValueNetwork,new_df)
            print("updated Network")
print(qValueNetwork.epsilon)
plt.plot(list(total_rewards["elm"]),list(total_rewards["elm"].values()))
plt.plot(list(total_rewards["random"]),list(total_rewards["random"].values()))        
plt.show()


# comparison={}
# for i in range(1):
#     comparison[i]={}
#     for j in new_df[11].unique():
#         comparison[i][j]={}
#         sub_df = new_df[new_df[11]==j]
#         current_state = np.array(sub_df[[0,1,2,3]])
#         next_state = np.array(sub_df[[5,6,7,8]])
#         rewards = np.array(sub_df[9])
#         q_pred = qValueNetwork.pred(x=next_state).detach().numpy()
#         q_indexes = np.argmax(q_pred,axis=1)
#         y_train = np.array([q_pred[i,q_indexes[i]]for i in range(len(q_pred))])*GAMMA+rewards
#         comparison[i][j]["1b"],comparison[i][j]["1a"],comparison[i][j]["2b"],comparison[i][j]["2a"]=qValueNetwork.train(accuracy=accuracy,n_iterations=iterations,x_train=current_state,y_train=y_train)

# fig,(ax1,ax2) = plt.subplots(1,2)
# b1 = []
# a1 = []
# b2 = []
# a2 = []
# for i in range(1):
#     b1_list = []
#     a1_list = []
#     b2_list = []
#     a2_list = []
#     for j in new_df[11].unique():   
#         b1_list.append(comparison[i][j]["1b"])
#         b2_list.append(comparison[i][j]["2b"])
#         a1_list.append(comparison[i][j]["1a"])
#         a2_list.append(comparison[i][j]["2a"])
    
#     b1.append(np.mean(b1_list))
#     a1.append(np.mean(b2_list))
#     b2.append(np.mean(a1_list))
#     a2.append(np.mean(a2_list))


#     ax1.plot(b1_list,label="before optim action 1")
#     ax1.plot(a1_list,label="after optim action 1")
#     ax2.plot(b2_list,label="before optim action 2")
#     ax2.plot(a2_list,label="after optim action 2")

# ax1.legend()
# ax2.legend()
# plt.show()