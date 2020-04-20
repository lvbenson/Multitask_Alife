# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 20:25:49 2020

@author: benso
"""


import numpy as np
import ffann                #Controller
import invpend              #Task 1
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
duration_IP = 10
stepsize_IP = 0.05
time_IP = np.arange(0.0,duration_IP,stepsize_IP)

MaxFit = 0.627 #Leggedwalker

# Fitness initialization ranges
#Inverted Pendulum
trials_theta_IP = 6
trials_thetadot_IP = 6
total_trials_IP = trials_theta_IP*trials_thetadot_IP
theta_range_IP = np.linspace(-np.pi, np.pi, num=trials_theta_IP)
thetadot_range_IP = np.linspace(-1.0,1.0, num=trials_thetadot_IP)


# Fitness function
def analysis(genotype):
    # Common setup
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    fitness = np.zeros(3)

    # Task 1
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
            for t in time_IP:
                nn.step(np.concatenate((body.state(),np.zeros(4),np.zeros(3)))) #arrays for inputs for each task
                nn_state_ip[k] = nn.states() #re-sets the activations for every k
                k += 1
                f += body.step(stepsize_IP, np.array([nn.output()[0]]))
            fit_IP[i][j] = ((f/duration_IP)+7.65)/7
            j += 1
        i += 1
    fitness[0] = np.mean(fit_IP)

    return fitness,fit_IP,nn_state_ip

gens = len(np.load(dir+"/average_history_T1_1.npy"))
gs=len(np.load(dir+"/best_individual_T1_1.npy"))
af = np.zeros((reps,gens))
bf = np.zeros((reps,gens))
bi = np.zeros((reps,gs))

fit_IP_list_good = []
fit_CP_list_good = []
fit_LW_list_good = []
fit_IP_list_bad = []
fit_CP_list_bad = []
fit_LW_list_bad = []
#best_net_list = []

best_list = []
for i in range(reps):
    af[i] = np.load(dir+"/average_history_T1_"+str(i)+".npy")
    bf[i] = np.load(dir+"/best_history_T1_"+str(i)+".npy")
    bi[i] = np.load(dir+"/best_individual_T1_"+str(i)+".npy")
    #best_net_list.append(bi[i])
    
    f,m1,ns1=analysis(bi[i])
    if bf[i][-1] > 0.8: #check last element in bf

        plt.plot(bf[i].T,'b')
        #if bf[i][-1] >=0.95:
            #plt.plot(bf[i].T,'r')
        
    #if bf[i][-1] < 0.8:
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
plt.title('evolution: IP Task')
plt.legend()
plt.savefig(dir+"/evolve_IP.png")
plt.show()