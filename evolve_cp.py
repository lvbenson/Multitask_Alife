# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 21:12:51 2020

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
duration_CP = 50
stepsize_CP = 0.05
time_CP = np.arange(0.0,duration_CP,stepsize_CP)

MaxFit = 0.627 #Leggedwalker

# Fitness initialization ranges

#Cartpole
trials_theta_CP = 4
trials_thetadot_CP = 4
trials_x_CP = 4
trials_xdot_CP = 4
total_trials_CP = trials_theta_CP*trials_thetadot_CP*trials_x_CP*trials_xdot_CP
theta_range_CP = np.linspace(-0.05, 0.05, num=trials_theta_CP)
thetadot_range_CP = np.linspace(-0.05, 0.05, num=trials_thetadot_CP)
x_range_CP = np.linspace(-0.05, 0.05, num=trials_x_CP)
xdot_range_CP = np.linspace(-0.05, 0.05, num=trials_xdot_CP)

# Fitness function
def analysis(genotype):
    # Common setup
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    fitness = np.zeros(3)

    
    # Task 2
    body = cartpole.Cartpole()
    nn_state_cp = np.zeros((total_trials_CP*len(time_CP),nI+nH1+nH2+nO))
    #total_steps = len(theta_range_CP) * len(thetadot_range_CP) * len(x_range_CP) * len(xdot_range_CP) * len(time_CP)
    fit_CP = np.zeros((len(theta_range_CP),len(thetadot_range_CP)))
    i = 0
    k = 0
    for theta in theta_range_CP:
        j = 0
        for theta_dot in thetadot_range_CP:
            f_cumulative = 0
            for x in x_range_CP:
                for x_dot in xdot_range_CP:
                    body.theta = theta
                    body.theta_dot = theta_dot
                    body.x = x
                    body.x_dot = x_dot
                    f = 0.0
                    for t in time_CP:
                        nn.step(np.concatenate((np.zeros(3),body.state(),np.zeros(3))))
                        nn_state_cp[k] = nn.states()
                        k += 1
                        f += body.step(stepsize_CP, np.array([nn.output()[1]]))
                    f_cumulative += f/duration_CP
                fit_CP[i][j] = f_cumulative/(len(x_range_CP)*len(xdot_range_CP))
            j += 1
        i += 1
    fitness[1] = np.mean(fit_CP)

   
    return fitness,fit_CP,nn_state_cp

gens = len(np.load(dir+"/average_history_T2_1.npy"))
gs=len(np.load(dir+"/best_individual_T2_1.npy"))
af = np.zeros((reps,gens))
bf = np.zeros((reps,gens))
bi = np.zeros((reps,gs))

best_list = []
for i in range(reps):
    af[i] = np.load(dir+"/average_history_T2_"+str(i)+".npy")
    bf[i] = np.load(dir+"/best_history_T2_"+str(i)+".npy")
    bi[i] = np.load(dir+"/best_individual_T2_"+str(i)+".npy")
    #best_net_list.append(bi[i])
    
    f,m1,ns1=analysis(bi[i])
    plt.plot(bf[i].T,'b')
    #if bf[i][-1] > 0.8: #check last element in bf

        #plt.plot(bf[i].T,'b')
        #if bf[i][-1] >=0.95:
            #plt.plot(bf[i].T,'r')
        
    #if bf[i][-1]<0.8:
        #plt.plot(af[i].T,'k')
        #plt.plot(bf[i].T,'y')
    #if bf[i][-1]>=0.95:
        #plt.plot(bf[i].T,'r')
        
    plt.plot(af[i].T,'y')
    best_list.append(bf[i][-1])
        
#print(best_list)
best_network = best_list.index(max(best_list))
print(best_network)


plt.plot(bf[best_network].T,'r',label='best network')
plt.xlabel('generations')
plt.ylabel('performance')
plt.title('evolution: CP Task')
plt.legend()
plt.savefig(dir+"/evolve_cp.png")
plt.show()