# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 20:25:49 2020

@author: benso
"""


import numpy as np
import ffann                #Controller
import invpend              #Task 1
import cartpole
import leggedwalker
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys

dir1 = str(sys.argv[1])
dir2 = str(sys.argv[2])
dir3 = str(sys.argv[3])
reps = int(sys.argv[4])

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

gens = len(np.load(dir1+"/average_history_T1_1.npy"))
gs=len(np.load(dir1+"/best_individual_T1_1.npy"))
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

plt.figure(figsize=(7,2.8))
plt.subplot(1,3,1)


best_list = []
for i in range(reps):
    af[i] = np.load(dir1+"/average_history_T1_"+str(i)+".npy")
    bf[i] = np.load(dir1+"/best_history_T1_"+str(i)+".npy")
    bi[i] = np.load(dir1+"/best_individual_T1_"+str(i)+".npy")
    #best_net_list.append(bi[i])
    
    f,m1,ns1=analysis(bi[i])
    #if bf[i][-1] > 0.8: #check last element in bf

    plt.plot(bf[i].T,'b',label='Best Fitness')
    
    plt.plot(af[i].T,'y',label='Average Fitness')    
    best_list.append(bf[i][-1])
        
#print(best_list)
best_network = best_list.index(max(best_list))
#print(best_network)

plt.plot(bf[best_network].T,'r',label='Best Network')
plt.xlabel('generations')
plt.ylabel('performance')
plt.title('(A)')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())


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
def analysis1(genotype):
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

gens = len(np.load(dir2+"/average_history_T2_1.npy"))
gs=len(np.load(dir2+"/best_individual_T2_1.npy"))
af = np.zeros((reps,gens))
bf = np.zeros((reps,gens))
bi = np.zeros((reps,gs))


plt.subplot(1,3,2)

best_list = []
for i in range(reps):
    af[i] = np.load(dir2+"/average_history_T2_"+str(i)+".npy")
    bf[i] = np.load(dir2+"/best_history_T2_"+str(i)+".npy")
    bi[i] = np.load(dir2+"/best_individual_T2_"+str(i)+".npy")
    
    f,m1,ns1=analysis1(bi[i])
    plt.plot(bf[i].T,'b')

        
    plt.plot(af[i].T,'y')
    best_list.append(bf[i][-1])
        
#print(best_list)
best_network = best_list.index(max(best_list))
#print(best_network)


plt.plot(bf[best_network].T,'r')
#plt.xlabel('generations')
#plt.ylabel('performance')
plt.title('(B)')

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
def analysis3(genotype):
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

gens = len(np.load(dir3+"/average_history_T3_1.npy"))
gs=len(np.load(dir3+"/best_individual_T3_1.npy"))
af = np.zeros((reps,gens))
bf = np.zeros((reps,gens))
bi = np.zeros((reps,gs))


plt.subplot(1,3,3)

best_list = []
for i in range(reps):
    af[i] = np.load(dir3+"/average_history_T3_"+str(i)+".npy")
    bf[i] = np.load(dir3+"/best_history_T3_"+str(i)+".npy")
    bi[i] = np.load(dir3+"/best_individual_T3_"+str(i)+".npy")
    #best_net_list.append(bi[i])
    
    f,m3,ns3=analysis3(bi[i])
    plt.plot(bf[i].T,'b')
    plt.plot(af[i].T,'y')
        
    best_list.append(bf[i][-1])
        
#print(best_list)
best_network = best_list.index(max(best_list))
#print(best_network)



plt.plot(bf[best_network].T,'r')
#plt.xlabel('generations')
#plt.ylabel('performance')
plt.title('(C)')


plt.tight_layout()
plt.savefig('EVOLVE_SEPARATE_NEW.png')
plt.show()


