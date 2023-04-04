"""
Created on Sun Mar 26 10:41:17 2023

@author: Mahshid Mansouri 
AE 598-Spring 2023-HW2: DQN implementation 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import random

    
## Define a class for the experience replay 
class ReplayMemory(object):
    
    """  
    Class for defining the experience replay memory 
    Args:
        capacity: Capacity of the memory for storing all transitions 
        batch_size: Batch size for sampling batch of data from the memory 
        
    Returns:
        The output of the Q-Network which is the Q-values corresponsing to an input state to the Q-Network and all possible actions that could be taken from that state  
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','done'))
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
   
    
## Define a class for the Q_Network 
class DQN(nn.Module):
    
    """  
    Class for defining the neural networks used for the Q-Network and the Target Network 
    Args:
        n_observations: Number of observations
        n_actions: Number of actions
        
    Returns:
        The output of the Q-Network which is the Q-values corresponsing to an input state to the Q-Network and all possible actions that could be taken from that state  
    """

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        y1 = torch.tanh(self.layer1(x))
        y2 = torch.tanh(self.layer2(y1))
        # y1 = F.tanh(self.layer1(x))
        # y2 = F.tanh(self.layer2(y1))
        output = self.layer3(y2)
        
        return output

def update_epsilon(epsilon, eps_min, eps_delta): 
    
    if epsilon >= eps_min: 
        epsilon = epsilon - eps_delta 
    else: 
        epsilon = eps_min 
    
    return epsilon 
    
    
## Define a function to select an epsilon-greedy action using a soft update rule 
def select_action(env, q_net, state, epsilon):
    
    """  
    Class for defining the neural networks used for the Q-Network and the Target Network 
    Args:
        n_observations: Number of observations
        n_actions: Number of actions
        
    Returns:
        The output of the Q-Network which is the Q-values corresponsing to an input state to the Q-Network and all possible actions that could be taken from that state  
    """

    if random.random() > epsilon:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return q_net(state).max(1)[1].view(1, 1)
    else:
        #return torch.tensor([[env.action_space.sample()]], device=None, dtype=torch.long)
        return torch.tensor([[random.randrange(env.num_actions)]], dtype=torch.long)


## The following function performs a single step of the optimization for the q_network
def optimize_Q(memory, BATCH_SIZE, q_net, target_net, gamma, optimizer):
    
    """  
    Optimizes the Q-Network 
    Args:
        memory: Replay memory buffer for storing the transitions 
        BATCH_SIZE: Batch size for sampling batch of data from the memory
        q_net = Q-Network 
        target_net: Target network 
        gamma: Discount factor 
        optimizer: Optimizer to be used for optimizing the Q-Network  
        
    Returns:
         
    """
    if len(memory) < BATCH_SIZE:
        return
    Transition = namedtuple('Transition',
                ('state', 'action', 'next_state', 'reward','done'))
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    
    ## Convert to tensor data type 
    state_batch = torch.stack(batch.state)[:,-1,:]
    action_batch = torch.stack(batch.action)[:,-1,:]
    reward_batch = torch.stack(batch.reward)


    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to q_net
    
    state_action_values = q_net(state_batch).gather(1,action_batch)
  
    
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    #if not batch.done:
        expected_state_action_values = (1-int(batch.done ==True))*(next_state_values * gamma) + reward_batch
    # else: 
    #     expected_state_action_values = reward_batch
    

    # Compute loss
    #criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the Q-Network 
    optimizer.zero_grad() #sets the gradients of the network parameters to zero before doing a backpropagation
    loss.backward() # performs a backpropagation of the loss w.r.t the network parameters
   
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(q_net.parameters(), 100)
    optimizer.step()
    return loss




        
            