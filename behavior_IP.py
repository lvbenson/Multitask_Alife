<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 14:56:10 2020

@author: benso
"""

import numpy as np
import ffann                #Controller
import invpend              #Task 1
import cartpole             #Task 2
import leggedwalker         #Task 3
import matplotlib.pyplot as plt
import sys

dir = str(sys.argv[1])
ind = int(sys.argv[2])

nI = 3+4+3
nH1 = 5
nH2 = 5
nO = 1+1+3 #output activation needs to account for 3 outputs in leggedwalker
WeightRange = 15.0
BiasRange = 15.0

    # Task Params
duration_IP = 10
stepsize_IP = 0.05
time_IP = np.arange(0.0,duration_IP,stepsize_IP)

  
trials_theta_IP = 10
trials_thetadot_IP = 10
total_trials_IP = trials_theta_IP*trials_thetadot_IP
theta_range_IP = np.linspace(-np.pi, np.pi, num=trials_theta_IP)
thetadot_range_IP = np.linspace(-1.0,1.0, num=trials_thetadot_IP)



# Fitness function

def analysis(genotype):
    # Common setup
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    #fitness = np.zeros(3)

    # Task 1
    theta_list = []
    body = invpend.InvPendulum()
    nn_state_ip = np.zeros((total_trials_IP*len(time_IP),nI+nH1+nH2+nO))
    total_steps = len(theta_range_IP) * len(thetadot_range_IP) * len(time_IP)
    fit_IP = np.zeros((len(theta_range_IP),len(thetadot_range_IP)))
    i=0
    k=0
    for theta in theta_range_IP:
        #print(len(theta_range_IP))
        j=0
        for theta_dot in thetadot_range_IP:
            body.theta = theta
            body.theta_dot = theta_dot
            f = 0.0
            th_behavior = []
            for t in time_IP:
                nn.step(np.concatenate((body.state(),np.zeros(4),np.zeros(3)))) #arrays for inputs for each task
                nn_state_ip[k] = nn.states() #re-sets the activations for every k
                k += 1
                f += body.step(stepsize_IP, np.array([nn.output()[0]]))
                #print('new thetas',((body.theta+np.pi) % (2*np.pi)) - np.pi)
                th = ((body.theta+np.pi) % (2*np.pi)) - np.pi
                th_behavior.append(th)
                
            
            fit_IP[i][j] = ((f/duration_IP)+7.65)/7
            j += 1
        i += 1
        theta_list.append(theta)
        plt.plot(th_behavior)
        plt.title("Inverted Pendulum Behavior")
        plt.xlabel("Time, 10s")
        plt.ylabel("Theta")
    plt.savefig(dir+"/IP_theta_"+str(ind)+".png")
    plt.show()
    
    plt.imshow(fit_IP,
               extent=(-3.141592653589793,3.141592653589793,-1,1),aspect='auto') #plotting fit_IP
    plt.colorbar()
    plt.xlabel("Theta")
    xticks = theta_list
    yticks = [-1.0,-0.7777777777777778,-0.5555555555555556,-0.33333333333333337,
              -0.11111111111111116,0.11111111111111116,0.33333333333333326,
              0.5555555555555554,0.7777777777777777,1.0]
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.ylabel("ThetaDot")
    plt.title("Inverted Pendulum")
    plt.savefig(dir+"/perfmap_IP_new_"+str(ind)+".png")
    plt.show()
      
    

bi = np.load(dir+"/best_individual_"+str(ind)+".npy")
analysis(bi)
        
        
        
=======
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 14:56:10 2020

@author: benso
"""

import numpy as np
import ffann                #Controller
import invpend              #Task 1
import cartpole             #Task 2
import leggedwalker         #Task 3
import matplotlib.pyplot as plt
import sys

dir = str(sys.argv[1])
ind = int(sys.argv[2])

nI = 3+4+3
nH1 = 5
nH2 = 5
nO = 1+1+3 #output activation needs to account for 3 outputs in leggedwalker
WeightRange = 15.0
BiasRange = 15.0

    # Task Params
duration_IP = 10
stepsize_IP = 0.05
time_IP = np.arange(0.0,duration_IP,stepsize_IP)

  
trials_theta_IP = 10
trials_thetadot_IP = 10
total_trials_IP = trials_theta_IP*trials_thetadot_IP
theta_range_IP = np.linspace(-np.pi, np.pi, num=trials_theta_IP)
thetadot_range_IP = np.linspace(-1.0,1.0, num=trials_thetadot_IP)



# Fitness function

def analysis(genotype):
    # Common setup
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    #fitness = np.zeros(3)

    # Task 1
    theta_list = []
    body = invpend.InvPendulum()
    nn_state_ip = np.zeros((total_trials_IP*len(time_IP),nI+nH1+nH2+nO))
    total_steps = len(theta_range_IP) * len(thetadot_range_IP) * len(time_IP)
    fit_IP = np.zeros((len(theta_range_IP),len(thetadot_range_IP)))
    i=0
    k=0
    for theta in theta_range_IP:
        #print(len(theta_range_IP))
        j=0
        for theta_dot in thetadot_range_IP:
            body.theta = theta
            body.theta_dot = theta_dot
            f = 0.0
            th_behavior = []
            for t in time_IP:
                nn.step(np.concatenate((body.state(),np.zeros(4),np.zeros(3)))) #arrays for inputs for each task
                nn_state_ip[k] = nn.states() #re-sets the activations for every k
                k += 1
                f += body.step(stepsize_IP, np.array([nn.output()[0]]))
                #print('new thetas',((body.theta+np.pi) % (2*np.pi)) - np.pi)
                th = ((body.theta+np.pi) % (2*np.pi)) - np.pi
                th_behavior.append(th)
                
            
            fit_IP[i][j] = ((f/duration_IP)+7.65)/7
            j += 1
        i += 1
        theta_list.append(theta)
        plt.plot(th_behavior)
        plt.title("Inverted Pendulum Behavior")
        plt.xlabel("Time, 10s")
        plt.ylabel("Theta")
    plt.savefig(dir+"/IP_theta_"+str(ind)+".png")
    plt.show()
    
    plt.imshow(fit_IP,
               extent=(-3.141592653589793,3.141592653589793,-1,1),aspect='auto') #plotting fit_IP
    plt.colorbar()
    plt.xlabel("Theta")
    xticks = theta_list
    yticks = [-1.0,-0.7777777777777778,-0.5555555555555556,-0.33333333333333337,
              -0.11111111111111116,0.11111111111111116,0.33333333333333326,
              0.5555555555555554,0.7777777777777777,1.0]
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.ylabel("ThetaDot")
    plt.title("Inverted Pendulum")
    plt.savefig(dir+"/perfmap_IP_new_"+str(ind)+".png")
    plt.show()
      
    

bi = np.load(dir+"/best_individual_"+str(ind)+".npy")
analysis(bi)
        
        
        
>>>>>>> e8fbe70b727e80f4fd5102d1e97bc3356029035b
        