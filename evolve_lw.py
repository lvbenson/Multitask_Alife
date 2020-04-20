# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 21:23:01 2020

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
reps = int(sys.argv[2])

# ANN Params
nI = 3+4+3
nH1 = 5
nH2 = 5
nO = 1+1+3 #output activation needs to account for 3 outputs in leggedwalker
WeightRange = 15.0
BiasRange = 15.0

# Task Params
duration_LW = 220.0
stepsize_LW = 0.1
time_LW = np.arange(0.0,duration_LW,stepsize_LW)

MaxFit = 0.627 #Leggedwalker

# Fitness initialization ranges

#Legged walker
trials_theta = 3
theta_range_LW = np.linspace(-np.pi/6, np.pi/6, num=trials_theta)
trials_omega_LW = 3
omega_range_LW = np.linspace(-1.0, 1.0, num=trials_omega_LW)
total_trials_LW = trials_theta * trials_omega_LW

# Fitness function
def analysis(genotype):
    # Common setup
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    fitness = np.zeros(3)

    # Task 1
    
    #Task 3
    body = leggedwalker.LeggedAgent(0.0,0.0)
    nn_state_lw = np.zeros((total_trials_LW*len(time_LW),nI+nH1+nH2+nO))
    #total_steps = len(theta_range_LW) * len(omega_range_LW) * len(time_LW)
    fit_LW = np.zeros((len(theta_range_LW),len(omega_range_LW)))
    i = 0
    k = 0
    for theta in theta_range_LW:
        j = 0
        for omega in omega_range_LW:
            body.reset()
            body.angle = theta
            body.omega = omega
            for t in time_LW:
                nn.step(np.concatenate((np.zeros(3),np.zeros(4),body.state())))
                nn_state_lw[k] = nn.states()
                k += 1
                body.step(stepsize_LW, np.array(nn.output()[2:5]))
            fit_LW[i][j] = (body.cx/duration_LW)/MaxFit
            j += 1
        i += 1
    fitness[2] = np.mean(fit_LW)
    return fitness,fit_LW,nn_state_lw

gens = len(np.load(dir+"/average_history_T3_1.npy"))
gs=len(np.load(dir+"/best_individual_T3_1.npy"))
af = np.zeros((reps,gens))
bf = np.zeros((reps,gens))
bi = np.zeros((reps,gs))


best_list = []
for i in range(reps):
    af[i] = np.load(dir+"/average_history_T3_"+str(i)+".npy")
    bf[i] = np.load(dir+"/best_history_T3_"+str(i)+".npy")
    bi[i] = np.load(dir+"/best_individual_T3_"+str(i)+".npy")
    #best_net_list.append(bi[i])
    
    f,m3,ns3=analysis(bi[i])
    plt.plot(bf[i].T,'b')
    plt.plot(af[i].T,'y')
        
    best_list.append(bf[i][-1])
        
#print(best_list)
best_network = best_list.index(max(best_list))
print(best_network)


plt.plot(bf[best_network].T,'r',label='best network')
plt.xlabel('generations')
plt.ylabel('performance')
plt.title('evolution: LW Task')
plt.legend()
plt.savefig(dir+"/evolve_lw.png")
plt.show()