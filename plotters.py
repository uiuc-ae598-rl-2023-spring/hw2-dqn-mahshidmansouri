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
    

def plot_state_value_function(env, policy, q_net):
    theta = np.linspace(-np.pi, np.pi,100)
    theta_dot = np.linspace(-env.max_thetadot,env.max_thetadot,100)
    x_axis, y_axis = np.meshgrid(theta, theta_dot)
    
    policy_array = np.zeros_like(x_axis)
    for i in range(len(theta)):
        for j in range(len(theta)):
            s = np.array((x_axis[i,j], y_axis[i,j]))
            policy_array[i,j] = policy(s)
      
    V_array = np.zeros_like(x_axis)
    
    for i in range(len(theta)):
        for j in range(len(theta)):
            s = np.array((x_axis[i,j], y_axis[i,j]))
            V_array[i,j] = torch.max(q_net(torch.from_numpy(s).float())).item()
    
    plt.figure(plt.gcf().number+1)
    plt.pcolor(x_axis, y_axis, V_array)
    plt.xlabel('theta')
    plt.ylabel('theta dot')
    plt.colorbar()
    plt.savefig('figures/state_value_function.png')

def plot_ablation_study_comparison(output_return_total, episode_num_total):
    """
    Plots the learning curve for all four cases in the ablation study 
    :param episode_num_total: number of episodes
    :param output_return_total: the return from all four cases 
    :return: figure
    """
    
    fig, ax = plt.subplots(1)
    fig.suptitle('Ablation study learning curve comparison')
    ax.plot(episode_num_total,output_return_total[0], label = "with replay, with target")
    ax.plot(episode_num_total,output_return_total[1], label = "with replay, without target")
    ax.plot(episode_num_total,output_return_total[2], label = "without replay, with target" )
    ax.plot(episode_num_total,output_return_total[3], label = "without replay, without target" )

    plt.xlabel("Number of episodes")
    plt.ylabel("Return") 
    plt.legend()
    plt.savefig("figures/ablation_study_learning_curve_comparison.png")
    
    