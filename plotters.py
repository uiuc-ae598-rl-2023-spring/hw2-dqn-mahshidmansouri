"""
Created on Sun Mar 26 10:41:17 2023

@author: Mahshid Mansouri 
AE 598-Spring 2023-HW2: DQN implementation 

"""
import math
import numpy as np
import torch 
import random
import matplotlib
import matplotlib.pyplot as plt
from helpers import TD_zero 


def plot_learning_curve(episode_num,output_return):
    """
    Plots the learning curve for at least one trained agent 
    :param episode_num: number of episodes
    :param output_return: the return 
    :return: figure
    """
    
    plt.plot(episode_num,output_return)
    plt.xlabel("Number of episodes")
    plt.ylabel("Return") 
    plt.title(r'Learning curve plot')
    plt.savefig("figures/learning_curve.png")
    
def plot_example_traj_pendulum(q_net, env):
    """
    Plots example trajectory of pendulum
    :param q_net: trained Q-Network 
    :param env: environment object
    :return: figure
    """   
    
    # Initialize simulation
    s = env.reset()

    # Create dict to store data from simulation
    data = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }

    # Simulate until episode is done
    done = False
    while not done:
        s = torch.Tensor(s)
        Q = q_net(s).detach().numpy()  
        a = np.argmax(Q)
        (s, r, done) = env.step(a)
        data['t'].append(data['t'][-1] + 1)
        data['s'].append(s)
        data['a'].append(a)
        data['r'].append(r)
        

    # Parse data from simulation
    data['s'] = np.array(data['s'])
    theta = data['s'][:, 0]
    thetadot = data['s'][:, 1]
    tau = [env._a_to_u(a) for a in data['a']]

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].plot(data['t'], theta, label='theta')
    ax[0].plot(data['t'], thetadot, label='thetadot')
    ax[0].legend()
    ax[1].plot(data['t'][:-1], tau, label='tau')
    ax[1].legend()
    ax[2].plot(data['t'][:-1], data['r'], label='r')
    ax[2].legend()
    ax[2].set_xlabel('time step')
    plt.tight_layout()
    plt.savefig('figures/example_trajectory.png')
    
    
 
def plot_state_value_function(env, q_net): 
    
    """
    Plots the leanred state-value function for the trained agent
    :param q_net: trained Q-Network 
    :param env: environment object
    :return: figure
    """   
    
    q_value = np.zeros((60,12))
    theta = np.arange(-3,3.5,0.5)
    theta_dot = np.arange(-15,15.5,0.5)

    for i in range(60):
        for j in range(12): 
            x = np.array([theta_dot[i], theta[j]])
            s = env._x_to_s(x)
            s = torch.Tensor(s)
            q_max = max(q_net(s).detach().numpy())
            q_value [i][j] = q_max        
    
    print(q_value)
    plt.pcolor(theta, theta_dot, q_value, cmap='RdBu')
    plt.title("State value function for the trained agent")
    plt.colorbar()     
    plt.xlabel("theta")
    plt.ylabel("theta dot")
    plt.savefig('figures/state_value_function.png')

