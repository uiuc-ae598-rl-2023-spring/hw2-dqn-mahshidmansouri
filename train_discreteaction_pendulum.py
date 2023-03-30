"""
Created on Sun Mar 26 10:41:17 2023

@author: Mahshid Mansouri 
AE 598-Spring 2023-HW2: DQN implementation 

"""
import numpy as np
import torch
import torch.optim as optim
from discreteaction_pendulum import Pendulum 

from helpers import ReplayMemory
from helpers import DQN
from helpers import select_action
from helpers import optimize_model
from helpers import map_state_to_number
from helpers import optimal_policy

from plotters import plot_example_traj_pendulum
from plotters import plot_learning_curve
from plotters import plot_state_value_function

## Define hyperparameters based on DQN paper from Mnih et al. 2015, Table 1
BATCH_SIZE = 32 # Number of transitions sampled from the replay buffer
gamma = 0.95 # Discount factor as mentioned in the previous section
EPS_START = 1 # Starting value of epsilon
EPS_END = 0.1 # Final value of epsilon
EPS_DECAY = 1000000 # Controls the rate of exponential decay of epsilon, higher means a slower decay
target_update_freq = 20 # Update frequency for the tagert netwrok weight updates 
LR = 0.00025 # Learning rate for the SGD optimization 

## Define the environment 
env = Pendulum()

# Get number of actions from gym action space
n_actions = env.num_actions

# Get the number of state observations
n_observations = env.num_states

# Define the Q-Network and target network using the DQN class definition 
q_net = DQN(n_observations, n_actions)
target_net = DQN(n_observations, n_actions)
target_net.load_state_dict(q_net.state_dict())

# Define the optimizer to be used in training the neural network 
#optimizer = optim.AdamW(q_net.parameters(), lr=LR, amsgrad=True)
optimizer = optim.RMSprop(q_net.parameters(),lr=0.00025, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.95)
                                
num_episodes = 1000 # Number of episodes          

with_TARGET_Q = True # Flag defined for the ablation study which will be turned on and off to include/exclude the target network 
with_REPLAY = True # Flag defined for the ablation study which will be turned on and off to include/exclude the replay buffer 



## Code for performing DQN algorithm 
def Train_DQN(num_episodes, gamma, with_TARGET_Q, with_REPLAY):
    
    # Define a memory buffer from the class of ReplayMemory to store all transitions 
    # If the ablation study includes the replay buffer (i.e., with_REPLAY = True),then we can assign the capacity to a certain value
    if (with_REPLAY): 
        capacity = 1000000
        memory = ReplayMemory(capacity)
        
    # If the ablation study does not include the replay buffer, the use the BATCH_SIZE as the capacity of the replay memory 
    else: 
        capacity = BATCH_SIZE 
        memory = ReplayMemory(capacity)
        
    output_return = [] # An array for storing the output return of each episode 
    episode_num = [] # An arrau for storing the eposide number 
   
    for i_episode in range(num_episodes):
                
        c = 0 # Parameter for updating the target network every C time steps
        r = 0 # Variable for storing the episode return 
        counter = 0 # Counter to be used as the gamma power in calculating the return
        
        done = False  # Flag to define if we reached the end of an episode
             
        # Initialize the environment and get the state
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        while not done:
            action = select_action(env, q_net, state, EPS_START, EPS_END, EPS_DECAY)
            observation, reward, done = env.step(action.item())
            reward = torch.tensor([reward])
            #done = terminated or truncated
    
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    
            # Store the transition in memory
            memory.push(state, action, next_state, reward)
    
            # Move to the next state
            state = next_state
            
            # Updating the respective vaLues
            r += gamma**counter * reward
            counter += 1
    
            # Perform one step of the optimization (on the Q-Network)
            optimize_model(memory, BATCH_SIZE, q_net, target_net, gamma, optimizer)
            #print(i_episode)
            
            # Soft update of the target network's weights
            # With replay, with target Q (i.e., the standard algorithm).
            if (with_TARGET_Q): 
                if (c % target_update_freq == 0):
                    target_net.state_dict = q_net.state_dict
                    
            # With replay, without target Q (i.e., the target network is reset after each step).
            else: 
                target_net.reset_parameters()
            c += 1
            #print(c)
         
        output_return.append(r)
        episode_num.append(i_episode)
    
    # Save the Q-Network Parameters
    torch.save(q_net.state_dict(), 'q_net_weights.pth')  
    print('Complete')
    
    return output_return, episode_num     


# Main code for outputting the results 
# To see the ablation study results, need to set with_TARGET_Q, with_REPLAY flags to True/False
def main(num_episodes, gamma, with_TARGET_Q, with_REPLAY): 
    
    ## Plot of the learning curve 
    output_return, episode_num = Train_DQN(num_episodes, gamma, with_TARGET_Q, with_REPLAY)
    plot_learning_curve(episode_num,output_return)
    
    
    ## A plot of an example trajectory for at least one trained agent, and 
    ## A plot of the policy for at least one trained agent.
    # Load the traiend Q-Network parameters to q_net   
    q_net = DQN(n_observations, n_actions)
    q_net.load_state_dict(torch.load('q_net_weights.pth'))
    plot_example_traj_pendulum(q_net, env)
   
    
    ## An animated gif of an example trajectory for at least one trained agent.
    # Define a policy that maps every state to the "zero torque" action
    policy = lambda s: np.argmax(q_net(torch.Tensor(s)).detach().numpy())
    
    # Simulate an episode and save the result as an animated gif
    env.video(policy, filename='figures/test_discreteaction_pendulum.gif')
    
    
    ## A plot of the state-value function for at least one trained agent.
    plot_state_value_function(env, q_net)

      
 