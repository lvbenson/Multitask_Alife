# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:43:09 2020

@author: benso
"""

import numpy as np
import ffann                #Controller
#import invpend              #Task 1
#import cartpole             #Task 2
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
duration_LW = 320.0
stepsize_LW = 0.1
time_LW = np.arange(0.0,duration_LW,stepsize_LW)

MaxFit = 0.627 #Leggedwalker

# Fitness initialization ranges

#Legged walker
trials_theta = 10
theta_range_LW = np.linspace(-np.pi/6, np.pi/6, num=trials_theta)
trials_omega_LW = 10
omega_range_LW = np.linspace(-1.0, 1.0, num=trials_omega_LW)
total_trials_LW = trials_theta * trials_omega_LW


def analysis(genotype):
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    
    
    body = leggedwalker.LeggedAgent(0.0,0.0)
    nn_state_lw = np.zeros((total_trials_LW*len(time_LW),nI+nH1+nH2+nO))
    #total_steps = len(theta_range_LW) * len(omega_range_LW) * len(time_LW)
    fit_LW = np.zeros((len(theta_range_LW),len(omega_range_LW)))
    i = 0
    k = 0
    
    theta_list = []
    for theta in theta_range_LW:
        j = 0
        omega_list = []
        for omega in omega_range_LW:
            body.reset()
            body.angle = theta
            body.omega = omega
            th_behavior = []
            for t in time_LW:
                nn.step(np.concatenate((np.zeros(3),np.zeros(4),body.state())))
                nn_state_lw[k] = nn.states()
                k += 1
                body.step(stepsize_LW, np.array(nn.output()[2:5]))
               #th = body.state()
               #th_new = th[1]
                th_new = body.angle
              # th_norm = ((th_new+np.pi) % (2*np.pi)) - np.pi
                th_behavior.append(th_new)
            plt.plot(th_behavior)
            fit_LW[i][j] = (body.cx/duration_LW)/MaxFit
            j += 1
            omega_list.append(omega)
        i += 1
        theta_list.append(theta)
        #print(len(th_behavior))
        #plt.plot(th_behavior)
    plt.title("Legged Walker Behavior")
    plt.xlabel("Time, 320s")
    plt.ylabel("Theta")
    plt.savefig(dir+"/LW_theta_"+str(ind)+".png")
    plt.show()
    
    plt.close()
    
    plt.imshow(fit_LW,
               extent=(-0.5235987755982988,0.5235987755982988,-1,1),
               aspect='auto')
    plt.colorbar()
    plt.xlabel("Theta")
    print(theta_list)
    xticks = theta_list
    yticks = omega_list
    #yticks = [-0.05,-0.030000000000000002,-0.010000000000000002,0.009999999999999995,0.03,0.05]
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.ylabel("omega")
    plt.title("Legged Walker")
    plt.savefig(dir+"/perfmap_LW_new_"+str(ind)+".png")
    plt.show()


bi = np.load(dir+"/best_individual_"+str(ind)+".npy")
analysis(bi)