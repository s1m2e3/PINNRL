import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward',"episode"))


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


def conver_to_lstm_data(data,sequence_length):
    data =np.array(data)
    new_shape = [data.shape[0]-sequence_length]
    new_shape.append(sequence_length)
    for i in np.arange(1,len(data.shape)):
        new_shape.append(data.shape[i])
    
    data_shape = tuple(new_shape)
    new_data = np.zeros(shape=data_shape)
    for i in range(len(data)-sequence_length):
        new_data[i]=data[i:i+sequence_length]
    
    return new_data


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

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        
        self.layer1 = nn.Linear(n_observations, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, n_actions)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=0.001)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        
        # x=torch.tensor(x,dtype=torch.float).reshape(len(x),25)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    def train(self,num_epochs,x,y):
        x=torch.tensor(x,dtype=torch.float).reshape(len(x),25)
        # print(y.shape)
        y=torch.tensor(y,dtype=torch.float)
        for i in range(num_epochs):
            pred = self.forward(x)
            self.optimizer.zero_grad()
            loss = self.criterion(pred,y)
            # loss -= torch.norm(pred)*0.001
            loss.backward()
            self.optimizer.step()
            # if i%100==0:
                # print(loss,"loss at epoch %i",i)
        print(loss,"loss at epoch %i",i)
        
env = gym.make("CartPole-v1")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
SEQUENCE_LENGTH = 5
# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
transition_net = DQN((n_observations+1)*SEQUENCE_LENGTH,n_observations).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model(count):
    if len(memory) < BATCH_SIZE:
        return
    index = random.randint(a=0,b=len(memory.memory)-1)
    episode_index = memory.memory[index].episode
    transitions_indexes = [memory.memory[i] for i in range(len(memory.memory)) if memory.memory[i].episode == episode_index]
    # transitions_indexes = memory.memory[transitions_indexes]
    transitions = memory.sample(BATCH_SIZE)
    
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    batch_indexes = Transition(*zip(*transitions_indexes))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    
    non_final_mask_indexes = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch_indexes.next_state)), device=device, dtype=torch.bool)
    non_final_next_states_indexes = torch.cat([s for s in batch_indexes.next_state
                                                if s is not None])


    indexes_not_none = [np.where(non_final_next_states_indexes.numpy()==non_final_next_states_indexes.numpy()[i,:])[0][0] for i in range(len(non_final_next_states_indexes))]
    test_index = np.arange(len(indexes_not_none))
    for i in range(len(indexes_not_none)):
        if indexes_not_none[i] != test_index[i]:
            break
    indexes_not_none=indexes_not_none[:i]
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_batch_indexes = torch.cat(batch_indexes.state)
    action_batch_indexes = torch.cat(batch_indexes.action)
    reward_batch_indexes = torch.cat(batch_indexes.reward)

    states_actions_indexes = torch.hstack((state_batch_indexes,action_batch_indexes))[indexes_not_none,:]
    # predicted_states = transition_net(states_actions)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    # print(SEQUENCE_LENGTH)
    
    states_actions = conver_to_lstm_data(states_actions_indexes,SEQUENCE_LENGTH)
    # non_final_next_states = conver_to_lstm_data(non_final_next_states_indexes,SEQUENCE_LENGTH-3)
    non_final_next_states_indexes = non_final_next_states_indexes[SEQUENCE_LENGTH+1:,:]
    
    # non_final_next_states=non_final_next_states[1+SEQUENCE_LENGTH:]
    # non_final_next_states =conver_to_lstm_data(non_final_next_states,SEQUENCE_LENGTH-3)
    if count %15 ==0:
        transition_net.train(100,states_actions,non_final_next_states_indexes)
        passed = torch.hstack((state_batch_indexes[0:5,:],action_batch_indexes[0:5]))
        
        for i in range(60):
            
            action = select_action(passed[0,0:4])
            new_state = transition_net()

    
if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 600
rewards={}
for i_episode in range(num_episodes):
    rewards[i_episode]=[]
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated
        rewards[i_episode].append(reward)
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward,i_episode)

        # Move to the next state
        state = next_state
        
        # Perform one step of the optimization (on the policy network)
        optimize_model(t)

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            # plot_durations()
            break
    print(i_episode)
# print('Complete')
# plot_durations(show_result=True)
# plt.ioff()
# plt.show()