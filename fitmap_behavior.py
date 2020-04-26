# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 16:14:15 2020

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
    
    plt.figure(figsize=(7.1,2.3))
    plt.subplot(1,3,1)
    
    plt.imshow(fit_IP,
               extent=(-3.141592653589793,3.141592653589793,-1,1),aspect='auto') #plotting fit_IP
    plt.colorbar()
    plt.xlabel("Theta")
    xticks = theta_list
    #yticks = [-1.0,-0.7777777777777778,-0.5555555555555556,-0.33333333333333337,
              #-0.11111111111111116,0.11111111111111116,0.33333333333333326,
              #0.5555555555555554,0.7777777777777777,1.0]
    yticks = [-1,1]
    plt.xticks(rotation=90)
    #plt.xticks(xticks)
    plt.yticks(yticks)
    plt.ylabel("ThetaDot")
    plt.title("(A)")
    plt.tight_layout()
    
    
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
        
    plt.subplot(1,3,2)    
    plt.imshow(fit_CP,
               extent=(-.05,0.05,-.05,0.05),aspect='auto') #plotting fit_IP
    plt.colorbar()
    plt.xlabel("Theta")
    #print(theta_list)
    xticks = theta_list
    yticks = theta_dot_list
    #yticks = [-0.05,-0.030000000000000002,-0.010000000000000002,0.009999999999999995,0.03,0.05]
    #plt.xticks(xticks)
    plt.xticks(rotation=90)
    yticks = [-0.05,0.05]
    plt.yticks(yticks)
    plt.ylabel("ThetaDot")
    plt.title("(B)")
    plt.tight_layout()
    
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
                th_norm = ((th_new+np.pi) % (2*np.pi)) - np.pi
                th_behavior.append(th_norm)
            #plt.plot(th_behavior)
            
            fit_LW[i][j] = (body.cx/duration_LW)/MaxFit
            j += 1
            omega_list.append(omega)
        i += 1
        theta_list.append(theta)
    
    plt.subplot(1,3,3)
    plt.imshow(fit_LW,
               extent=(-0.5235987755982988,0.5235987755982988,-1,1),
               aspect='auto')
    plt.colorbar()
    plt.xlabel("Theta")
    #print(theta_list)
    xticks = theta_list
    
    yticks = [-1,1]
    #yticks = [-0.05,-0.030000000000000002,-0.010000000000000002,0.009999999999999995,0.03,0.05]
    #plt.xticks(xticks)
    plt.xticks(rotation=90)
    plt.yticks(yticks)
    plt.ylabel("omega")
    plt.title("(C)")
    plt.tight_layout()
    plt.savefig('all_perf_behavior.png')
    plt.show()
    
bi = np.load(dir+"/best_individual_"+str(ind)+".npy")
analysis(bi)