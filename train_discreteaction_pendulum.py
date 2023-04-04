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
from helpers import update_epsilon
from helpers import select_action
from helpers import optimize_Q

from plotters import plot_example_traj_pendulum
from plotters import plot_learning_curve
from plotters import plot_state_value_function
from plotters import plot_ablation_study_comparison


## Define hyperparameters based on DQN paper from Mnih et al. 2015, Table 1
BATCH_SIZE = 32 # Number of transitions sampled from the replay buffer
gamma = 0.95 # Discount factor as mentioned in the previous section
eps_max = 1 # Starting value of epsilon
eps_min = 0.1 # Final value of epsilon
eps_delta = 0.0001 # Controls the rate of exponential decay of epsilon, higher means a slower decay
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
#optimizer = optim.RMSprop(q_net.parameters(),lr=0.00025, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.95)
optimizer = optim.RMSprop(q_net.parameters(),lr=0.00025)         
num_episodes = 200 # Number of episodes          

with_TARGET_Q = True # Flag defined for the ablation study which will be turned on and off to include/exclude the target network 
with_REPLAY = True # Flag defined for the ablation study which will be turned on and off to include/exclude the replay buffer 

# Initialize epsilon to be eps_max
epsilon = eps_max
l = []

## Code for performing DQN algorithm 
def Train_DQN(num_episodes, gamma, epsilon, with_REPLAY, with_TARGET_Q):
    
    # If the ablation study includes the replay buffer (i.e., with_REPLAY = True),then we can assign the capacity to a certain value
    if (with_REPLAY): 
        # Define a memory buffer from the class of ReplayMemory to store all transitions 
        capacity = 10000000
        memory = ReplayMemory(capacity)
        
    # If the ablation study does not include the replay buffer, the use the BATCH_SIZE as the capacity of the replay memory 
    else: 
        capacity = BATCH_SIZE 
        memory = ReplayMemory(capacity)
        
    output_return = [] # An array for storing the output return of each episode 
    episode_num = [] # An array for storing the eposide number 
   
    for i_episode in range(num_episodes):
                
        r = 0 # Variable for storing the episode return 
        counter = 0 # Counter to be used as the gamma power in calculating the return
        
        done = False  # Flag to define if we reached the end of an episode
             
        # Initialize the environment and get the state
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        while not done:
            action = select_action(env, q_net, state, epsilon)
            observation, reward, done = env.step(action.item())
            reward = torch.tensor([reward])
            done = torch.tensor([done])
    
            if done:
                next_state = None
                # Store the transition in memory
                memory.push(state, action, next_state, reward, done)
            else:
                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    
            # Store the transition in memory
            memory.push(state, action, next_state, reward, done)
    
            # Move to the next state
            state = next_state
            
            # Updating the respective vaLues
            r += (gamma**counter) * reward
            counter += 1
            
            if done: 
                break
    
            # Perform one step of the optimization (on the Q-Network)
            loss = optimize_Q(memory, BATCH_SIZE, q_net, target_net, gamma, optimizer)
            l.append(loss)
            # Store the transition in memory
            memory.push(state, action, next_state, reward, done)
            
            # Decay epsilon 
            epsilon = update_epsilon(epsilon, eps_min, eps_delta)
            
            # With replay, without target Q (i.e., the target network weights are updated after each step).
            if (with_TARGET_Q == False):
                target_net.load_state_dict(q_net.state_dict())
                
        # Soft update of the target network's weights after each "target_update_freq" episodes
        # With replay, with target Q (i.e., the standard algorithm).  
        if (with_TARGET_Q): 
            if (i_episode % target_update_freq == 0):
                print("targte updated", i_episode, epsilon)
                target_net.load_state_dict(q_net.state_dict())
            
         
        output_return.append(r)
        episode_num.append(i_episode)
    
    # Save the Q-Network Parameters
    torch.save(q_net.state_dict(), 'q_net_weights.pth')  
    print('Complete')
    
    return output_return, episode_num, l     


# Main code for outputting the results 
# To see the ablation study results, need to set with_TARGET_Q, with_REPLAY flags to True/False
def main(num_episodes, gamma, with_TARGET_Q, with_REPLAY): 
    
       
    if ((with_REPLAY == True) & (with_TARGET_Q == True)):
        ## Plot of the learning curve 
        output_return_1, episode_num_1 = Train_DQN(num_episodes, gamma, epsilon, with_REPLAY, with_TARGET_Q)
        plot_learning_curve(episode_num_1,output_return_1)
        
        
        ## A plot of an example trajectory for at least one trained agent, and 
        ## A plot of the policy for at least one trained agent.
        # Load the traiend Q-Network parameters to q_net   
        q_net_1 = DQN(n_observations, n_actions)
        q_net_1.load_state_dict(torch.load('q_net_weights_1.pth'))
        plot_example_traj_pendulum(q_net_1, env)
       
        
        ## An animated gif of an example trajectory for at least one trained agent.
        # Define a policy that maps every state to the "zero torque" action
        policy_1 = lambda s: np.argmax(q_net_1(torch.Tensor(s)).detach().numpy())
        
        # Simulate an episode and save the result as an animated gif
        env.video(policy_1, filename='figures/test_discreteaction_pendulum_1.gif')
        
        
        ## A plot of the state-value function for at least one trained agent.
        plot_state_value_function(env, policy_1, q_net_1)

    if ((with_REPLAY == True) & (with_TARGET_Q == False)):
        ## Plot of the learning curve 
        output_return_2, episode_num_2 = Train_DQN(num_episodes, gamma, epsilon, with_REPLAY, with_TARGET_Q)
        plot_learning_curve(episode_num_2,output_return_2)
        
        
        ## A plot of an example trajectory for at least one trained agent, and 
        ## A plot of the policy for at least one trained agent.
        # Load the traiend Q-Network parameters to q_net   
        q_net_2 = DQN(n_observations, n_actions)
        q_net_2.load_state_dict(torch.load('q_net_weights_2.pth'))
        plot_example_traj_pendulum(q_net_2, env)
       
        
        ## An animated gif of an example trajectory for at least one trained agent.
        # Define a policy that maps every state to the "zero torque" action
        policy_2 = lambda s: np.argmax(q_net_2(torch.Tensor(s)).detach().numpy())
        
        # Simulate an episode and save the result as an animated gif
        env.video(policy_2, filename='figures/test_discreteaction_pendulum_2.gif')
        
        
        ## A plot of the state-value function for at least one trained agent.
        plot_state_value_function(env, policy_2, q_net_2)
        
    if ((with_REPLAY == False) & (with_TARGET_Q == True)):
        ## Plot of the learning curve 
        output_return_3, episode_num_3 = Train_DQN(num_episodes, gamma, epsilon, with_REPLAY, with_TARGET_Q)
        plot_learning_curve(episode_num_3,output_return_3)
        
        
        ## A plot of an example trajectory for at least one trained agent, and 
        ## A plot of the policy for at least one trained agent.
        # Load the traiend Q-Network parameters to q_net   
        q_net_3 = DQN(n_observations, n_actions)
        q_net_3.load_state_dict(torch.load('q_net_weights_3.pth'))
        plot_example_traj_pendulum(q_net_3, env)
       
        
        ## An animated gif of an example trajectory for at least one trained agent.
        # Define a policy that maps every state to the "zero torque" action
        policy_3 = lambda s: np.argmax(q_net_1(torch.Tensor(s)).detach().numpy())
        
        # Simulate an episode and save the result as an animated gif
        env.video(policy_3, filename='figures/test_discreteaction_pendulum_3.gif')
        
        
        ## A plot of the state-value function for at least one trained agent.
        plot_state_value_function(env, policy_3, q_net_3)
        
    if ((with_REPLAY == False) & (with_TARGET_Q == False)):
        ## Plot of the learning curve 
        output_return_4, episode_num_4 = Train_DQN(num_episodes, gamma, epsilon, with_REPLAY, with_TARGET_Q)
        plot_learning_curve(episode_num_4,output_return_4)
        
        
        ## A plot of an example trajectory for at least one trained agent, and 
        ## A plot of the policy for at least one trained agent.
        # Load the traiend Q-Network parameters to q_net   
        q_net_4 = DQN(n_observations, n_actions)
        q_net_4.load_state_dict(torch.load('q_net_weights_4.pth'))
        plot_example_traj_pendulum(q_net_4, env)
       
        
        ## An animated gif of an example trajectory for at least one trained agent.
        # Define a policy that maps every state to the "zero torque" action
        policy_4 = lambda s: np.argmax(q_net_4(torch.Tensor(s)).detach().numpy())
        
        # Simulate an episode and save the result as an animated gif
        env.video(policy_4, filename='figures/test_discreteaction_pendulum_4.gif')
        
        
        ## A plot of the state-value function for at least one trained agent.
        plot_state_value_function(env, policy_4, q_net_4)
        
    # Ablation study: plot of the learning curve four all four cases
    output_return_total = [output_return_1[0:150], output_return_2[0:150], output_return_3[0:150], output_return_4[0:150]]
    episode_num_total = episode_num_1[0:150]
    plot_ablation_study_comparison(output_return_total, episode_num_total)
    
       
