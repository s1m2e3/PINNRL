from model_dqn import *
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
import seaborn as sns

sns.set_theme()

def predictFuture(qNetwork,transferNetwork,state):
    actions = []
    time = np.arange(50)
    main_state = state
    for i in range(200):
        new_states_list = []
        reward_list = []
        states_list = []
        for i in range(50):
            states_list.append(state)
            pred = qValueNetwork.pred(x=state).detach()
            actionPrediction = np.argmax(pred[:,0].numpy())
            new_states = transferNetwork.pred([i,actionPrediction]).flatten().detach().numpy()
            reward = customReward(new_states)
            new_states_list.append(new_states)
            reward_list.append(reward)
            state=new_states
            
        new_states_list = np.array(new_states_list)
        reward_list = np.array(reward_list)
        states_list = np.array(states_list)
        
        current_state = states_list
        next_state = new_states_list
        rewards = reward_list
        q_pred = qNetwork.pred(x=next_state).detach().numpy()
        q_indexes = np.argmax(q_pred,axis=1)
        y_train = np.array([q_pred[i,q_indexes[i]]for i in range(len(q_pred))])*GAMMA+rewards
        _,_,_,_=qValueNetwork.train(accuracy=1,n_iterations=1,x_train=current_state,y_train=y_train)

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
    return (goalX+goalDX+goalTheta+goalDTheta)/50

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

n_nodes = 256
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
simEpisodes = 500
simTrainings = 100

# Get number of actions from gym action space
n_actions = env.action_space.n

# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

episode_durations = []
data = {}
total_rewards={"DQN":{},"random":{},"DQN proxy":{}}
plt.figure()
for name in ["DQN","random"]:

    
    qValueNetworkPolicy = NN(hidden_size=n_nodes,input_size1=n_observations,output_size=n_actions)
    qValueNetworkTarget = NN(hidden_size=n_nodes,input_size1=n_observations,output_size=n_actions)
    transferFunctionNetwork = NN(hidden_size=n_nodes,input_size1=n_actions+n_observations,output_size=n_observations)
    memory = ReplayMemory(10000)
    count = 0
    for i in range(simEpisodes):
        
        done = False
        state,info = env.reset()
        data[i]={"state":[],"action":[],"next_state":[],"reward":[]}
        
        # if i % 20==0 and i>0 and name == "elm proxy":
        #     qValueNetwork=predictFuture(qValueNetwork,transferFunctionNetwork,state)
    
        
        while not done:
            if qValueNetworkPolicy.epsilon<random.random() and (name =="DQN" or name =="elm proxy"):
                pred = qValueNetworkPolicy.forward(x=state).detach()
                actionPrediction = np.argmax(pred.numpy())
                
            
            else: 
            
                actionPrediction = random.choice([0,1])
                qValueNetworkPolicy.epsilon -= qValueNetworkPolicy.epsilon_decay

            next_state, reward, terminated, truncated, _ = env.step(actionPrediction)
            memory.push(torch.tensor(state), torch.tensor(actionPrediction).unsqueeze(0), torch.tensor(next_state), torch.tensor(reward).unsqueeze(0),count,i)
            if count>BATCH_SIZE:
                transitions=memory.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))

                # Compute a mask of non-final states and concatenate the batch elements
                # (a final state would've been the one after which simulation ended)
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=device, dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch.next_state
                                                            if s is not None]).reshape(BATCH_SIZE,4)
                state_batch = torch.cat(batch.state).reshape(BATCH_SIZE,4)
                action_batch = torch.cat(batch.action).reshape(BATCH_SIZE,1)
                reward_batch = torch.cat(batch.reward).reshape(BATCH_SIZE,1)

                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                # columns of actions taken. These are the actions which would've been taken
                # for each batch state according to policy_net
                state_action_values = qValueNetworkPolicy(state_batch).gather(1, action_batch)
                # state_action_values = qValueNetworkPolicy(state_batch)
                
                # Compute V(s_{t+1}) for all next states.
                # Expected values of actions for non_final_next_states are computed based
                # on the "older" target_net; selecting their best reward with max(1)[0].
                # This is merged based on the mask, such that we'll have either the expected
                # state value or 0 in case the state was final.
                next_state_values = torch.zeros(BATCH_SIZE, device=device)
                
                with torch.no_grad():
                    next_state_values[non_final_mask] = qValueNetworkTarget(non_final_next_states).max(1)[0]
                next_state_values=next_state_values.reshape(BATCH_SIZE,1)
                # Compute the expected Q values
                expected_state_action_values = (next_state_values * GAMMA) + reward_batch
                qValueNetworkPolicy.train(1,x_train_data=state_batch,y_train_data=expected_state_action_values)
                
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                qValueNetworkTargetDict = qValueNetworkTarget.state_dict()
                qValueNetworkPolicyDict = qValueNetworkPolicy.state_dict()
                for key in qValueNetworkPolicyDict:
                    qValueNetworkTargetDict[key] = qValueNetworkPolicyDict[key]*TAU + qValueNetworkTargetDict[key]*(1-TAU)
                qValueNetworkTarget.load_state_dict(qValueNetworkTargetDict)
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
        
        # if  i%20==0 and i>0:
        #     episodes = list(data)
        #     passed_data = episodes[-20:]
        #     res = dict((k, data[k]) for k in passed_data if k in data)
        #     df = pd.DataFrame.from_dict(res).T
        #     accuracy = 1e-4
        #     iterations = 1
        #     new_df = pd.DataFrame()
        #     new_df = extract(df,new_df)
            
        #     for i in new_df[11].unique():
        #         sub_df = new_df[new_df[11]==i]
        #         
            # qValueNetwork=predictFuture(qValueNetwork,new_df)

plt.plot(list(total_rewards["DQN"]),list(total_rewards["DQN"].values()))
plt.plot(list(total_rewards["random"]),list(total_rewards["random"].values()))        
# plt.plot(list(total_rewards["elm proxy"]),list(total_rewards["elm proxy"].values()))        
# plt.legend(["ELM","random walk","ELM Proxy"])
plt.legend(["DQN","random walk"])
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