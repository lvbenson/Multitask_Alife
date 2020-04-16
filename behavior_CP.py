<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:18:14 2020

@author: benso
"""

import numpy as np
import ffann                #Controller
#import invpend              #Task 1
import cartpole             #Task 2
#import leggedwalker         #Task 3
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
duration_CP = 10
stepsize_CP = 0.05
time_CP = np.arange(0.0,duration_CP,stepsize_CP)

#Cartpole
trials_theta_CP = 7
trials_thetadot_CP = 7
trials_x_CP = 4
trials_xdot_CP = 4
total_trials_CP = trials_theta_CP*trials_thetadot_CP*trials_x_CP*trials_xdot_CP
theta_range_CP = np.linspace(-0.05, 0.05, num=trials_theta_CP)
thetadot_range_CP = np.linspace(-0.05, 0.05, num=trials_thetadot_CP)
x_range_CP = np.linspace(-0.05, 0.05, num=trials_x_CP)
xdot_range_CP = np.linspace(-0.05, 0.05, num=trials_xdot_CP)


def analysis(genotype):
    # Common setup
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    #fitness = np.zeros(3)
    
    
    theta_list = []
    body = cartpole.Cartpole()
    nn_state_cp = np.zeros((total_trials_CP*len(time_CP),nI+nH1+nH2+nO))
    #total_steps = len(theta_range_CP) * len(thetadot_range_CP) * len(x_range_CP) * len(xdot_range_CP) * len(time_CP)
    fit_CP = np.zeros((len(theta_range_CP),len(thetadot_range_CP)))
    i = 0
    k = 0
    for theta in theta_range_CP:
        #print(theta)
        j = 0
        theta_dot_list = []
        for theta_dot in thetadot_range_CP:
            f_cumulative = 0
            for x in x_range_CP:
                for x_dot in xdot_range_CP:
                    body.theta = theta
                    body.theta_dot = theta_dot
                    body.x = x
                    body.x_dot = x_dot
                    f = 0.0
                    th_behavior = []
                    for t in time_CP:
                        nn.step(np.concatenate((np.zeros(3),body.state(),np.zeros(3))))
                        nn_state_cp[k] = nn.states()
                        k += 1
                        f += body.step(stepsize_CP, np.array([nn.output()[1]]))
                        th = ((body.theta+np.pi) % (2*np.pi)) - np.pi
                        th_behavior.append(th)
                    f_cumulative += f/duration_CP
                    
                fit_CP[i][j] = f_cumulative/(len(x_range_CP)*len(xdot_range_CP))
            j += 1
            theta_dot_list.append(theta_dot)
        i += 1
        
        theta_list.append(theta)
        plt.plot(th_behavior)
        plt.title("Cartpole Behavior")
        plt.xlabel("Time, 10s")
        plt.ylabel("Theta")
    plt.savefig(dir+"/CP_theta_"+str(ind)+".png")
    plt.show()
    plt.close()
    
    plt.imshow(fit_CP,
               extent=(-.05,0.05,-.05,0.05),aspect='auto') #plotting fit_IP
    plt.colorbar()
    plt.xlabel("Theta")
    print(theta_list)
    xticks = theta_list
    yticks = theta_dot_list
    #yticks = [-0.05,-0.030000000000000002,-0.010000000000000002,0.009999999999999995,0.03,0.05]
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.ylabel("ThetaDot")
    plt.title("CartPole")
    plt.savefig(dir+"/perfmap_CP_new_"+str(ind)+".png")
    plt.show()


bi = np.load(dir+"/best_individual_"+str(ind)+".npy")
=======
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:18:14 2020

@author: benso
"""

import numpy as np
import ffann                #Controller
#import invpend              #Task 1
import cartpole             #Task 2
#import leggedwalker         #Task 3
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
duration_CP = 10
stepsize_CP = 0.05
time_CP = np.arange(0.0,duration_CP,stepsize_CP)

#Cartpole
trials_theta_CP = 7
trials_thetadot_CP = 7
trials_x_CP = 4
trials_xdot_CP = 4
total_trials_CP = trials_theta_CP*trials_thetadot_CP*trials_x_CP*trials_xdot_CP
theta_range_CP = np.linspace(-0.05, 0.05, num=trials_theta_CP)
thetadot_range_CP = np.linspace(-0.05, 0.05, num=trials_thetadot_CP)
x_range_CP = np.linspace(-0.05, 0.05, num=trials_x_CP)
xdot_range_CP = np.linspace(-0.05, 0.05, num=trials_xdot_CP)


def analysis(genotype):
    # Common setup
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    #fitness = np.zeros(3)
    
    
    theta_list = []
    body = cartpole.Cartpole()
    nn_state_cp = np.zeros((total_trials_CP*len(time_CP),nI+nH1+nH2+nO))
    #total_steps = len(theta_range_CP) * len(thetadot_range_CP) * len(x_range_CP) * len(xdot_range_CP) * len(time_CP)
    fit_CP = np.zeros((len(theta_range_CP),len(thetadot_range_CP)))
    i = 0
    k = 0
    for theta in theta_range_CP:
        #print(theta)
        j = 0
        theta_dot_list = []
        for theta_dot in thetadot_range_CP:
            f_cumulative = 0
            for x in x_range_CP:
                for x_dot in xdot_range_CP:
                    body.theta = theta
                    body.theta_dot = theta_dot
                    body.x = x
                    body.x_dot = x_dot
                    f = 0.0
                    th_behavior = []
                    for t in time_CP:
                        nn.step(np.concatenate((np.zeros(3),body.state(),np.zeros(3))))
                        nn_state_cp[k] = nn.states()
                        k += 1
                        f += body.step(stepsize_CP, np.array([nn.output()[1]]))
                        th = ((body.theta+np.pi) % (2*np.pi)) - np.pi
                        th_behavior.append(th)
                    f_cumulative += f/duration_CP
                    
                fit_CP[i][j] = f_cumulative/(len(x_range_CP)*len(xdot_range_CP))
            j += 1
            theta_dot_list.append(theta_dot)
        i += 1
        
        theta_list.append(theta)
        plt.plot(th_behavior)
        plt.title("Cartpole Behavior")
        plt.xlabel("Time, 10s")
        plt.ylabel("Theta")
    plt.savefig(dir+"/CP_theta_"+str(ind)+".png")
    plt.show()
    plt.close()
    
    plt.imshow(fit_CP,
               extent=(-.05,0.05,-.05,0.05),aspect='auto') #plotting fit_IP
    plt.colorbar()
    plt.xlabel("Theta")
    print(theta_list)
    xticks = theta_list
    yticks = theta_dot_list
    #yticks = [-0.05,-0.030000000000000002,-0.010000000000000002,0.009999999999999995,0.03,0.05]
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.ylabel("ThetaDot")
    plt.title("CartPole")
    plt.savefig(dir+"/perfmap_CP_new_"+str(ind)+".png")
    plt.show()


bi = np.load(dir+"/best_individual_"+str(ind)+".npy")
>>>>>>> e8fbe70b727e80f4fd5102d1e97bc3356029035b
analysis(bi)