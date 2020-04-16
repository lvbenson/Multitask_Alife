# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 19:47:47 2020

@author: benso
"""

import numpy as np
import matplotlib.pyplot as plt
import sys


dir = str(sys.argv[1])
reps = int(sys.argv[2])

for ind in range(reps):
    bf = np.load(dir+"/best_history_"+str(ind)+".npy")
    if bf[-1]>0.8:
        ip_lesion = np.load("./{}/lesions_IP_{}.npy".format(dir,ind))
        cp_lesion = np.load("./{}/lesions_CP_{}.npy".format(dir,ind))
        lw_lesion = np.load("./{}/lesions_LW_{}.npy".format(dir,ind))
        no_task_list = []
        ip_task_list = []
        cp_task_list = []
        lw_task_list = []
        ip_cp_task_list = []
        ip_lw_task_list = []
        cp_lw_task_list = []
        all_task_list = []
        
        Threshold = 0.99
        for (ip_neuron, cp_neuron, lw_neuron) in zip(ip_lesion,cp_lesion,lw_lesion):
            if ip_neuron > Threshold and cp_neuron > Threshold and lw_neuron > Threshold: # no task neurons
                no_task_list.append(1)
            if ip_neuron <= Threshold and cp_neuron > Threshold and lw_neuron > Threshold: # ip task neurons
                ip_task_list.append(1)
            if ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron > Threshold: # cp task neurons
                cp_task_list.append(1)
            if ip_neuron > Threshold and cp_neuron > Threshold and lw_neuron <= Threshold: #lw task neurons
                lw_task_list.append(1)
            if ip_neuron <= Threshold and cp_neuron <= Threshold and lw_neuron > Threshold: # ip + cp task neurons
                ip_cp_task_list.append(1)
            if ip_neuron <= Threshold and cp_neuron > Threshold and lw_neuron <= Threshold: # ip + lw task neurons
                ip_lw_task_list.append(1)
            if ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # cp + lw task neurons
                cp_lw_task_list.append(1)
            if ip_neuron <=  Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # all  task neurons
                all_task_list.append(1)

    
#8 separate lists of tuples (for categories)
#tuples: (network, number of neurons in that category)

(ind, len(no_task_list))

(ind, len(ip_task_list))


                
