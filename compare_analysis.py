# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 09:40:08 2020

@author: Lauren Benson
"""

import numpy as np
import matplotlib.pyplot as plt
import sys


#viz to compare: (1) variance against lesions. (2) mutual information against lesions.
# Lesions = "ground truth"

dir = str(sys.argv[1])
ind = int(sys.argv[2])
viz = int(sys.argv[3])

#load lesions data
ip_lesion = np.load("./{}/Lesions_IP_{}.npy".format(dir,ind))
#print(ip_lesion[0])
cp_lesion = np.load("./{}/Lesions_CP_{}.npy".format(dir,ind))
lw_lesion = np.load("./{}/Lesions_LW_{}.npy".format(dir,ind))

#load variance data
ip_var = np.load("./{}/Var_IP_{}.npy".format(dir,ind))
cp_var = np.load("./{}/Var_CP_{}.npy".format(dir,ind))
lw_var = np.load("./{}/Var_LW_{}.npy".format(dir,ind))

#load mutual information data
# =============================================================================
# ip_mi = np.load("./{}/MI_IP_{}.npy".format(dir,ind))
# cp_mi = np.load("./{}/MI_CP_{}.npy".format(dir,ind))
# lw_mi = np.load("./{}/MI_LW_{}.npy".format(dir,ind))
# 
# =============================================================================

#plot performance vs neurons for all three methods. Compare.

if viz == 1:
    #plot lesions
    plt.plot(ip_lesion, color="red",marker = "o", linestyle = 'none')
    plt.plot(cp_lesion,color="green",marker = "o",linestyle = 'none')
    plt.plot(lw_lesion,color="blue",marker = "o",linestyle = 'none')
    #plot variance
    plt.plot(ip_var, color="red",marker = "^",linestyle = 'none')
    plt.plot(cp_var, color="green",marker = "^",linestyle = 'none')
    plt.plot(lw_var, color="blue",marker = "^",linestyle = 'none')
    #plot mutual information 
    #plt.plot(ip_mi, color="red",marker = "s", linestyle = 'none')
    #plt.plot(cp_mi, color="red",marker = "s", linestyle = 'none')
    #plt.plot(lw_mi, color="red",marker = "s", linestyle = 'none')
    plt.xlabel("Interneuron")
    plt.ylabel("Performance")
    plt.title("Lesions vs Variance vs MutualInfo")
    plt.savefig(dir+"/Compare_"+str(ind)+".png")
    plt.show()
    plt.close()

#Finer-grain comparison
#variance vs infolesion
#10 neurons, 2 hidden layers 


    figure, axes = plt.subplots(nrows=2, ncols=5)

    axes[0, 0].plot(ip_lesion[5],color = "purple",marker="o")
    axes[0, 0].plot(ip_var[5],color="yellow",marker="^")
    axes[0,0].title.set_text('HL 1, Neuron 1')
    
    axes[0,1].plot(ip_lesion[6],color = "purple",marker="o")
    axes[0,1].plot(ip_var[6],color="yellow",marker="^")
    axes[0,1].title.set_text('HL 1, Neuron 2')
    
    axes[0,2].plot(ip_lesion[7],color = "purple",marker="o")
    axes[0,2].plot(ip_var[7],color="yellow",marker="^")
    axes[0,2].title.set_text('HL 1, Neuron 3')
    
    axes[0,3].plot(ip_lesion[8],color = "purple",marker="o")
    axes[0,3].plot(ip_var[8],color="yellow",marker="^")
    axes[0,3].title.set_text('HL 1, Neuron 4')
    
    axes[0,4].plot(ip_lesion[9],color = "purple",marker="o")
    axes[0,4].plot(ip_var[9],color="yellow",marker="^")
    axes[0,4].title.set_text('HL 1, Neuron 5')
    
    axes[1,0].plot(ip_lesion[0],color = "purple",marker="o")
    axes[1,0].plot(ip_var[0],color="yellow",marker="^")
    axes[1,0].title.set_text('HL 2, Neuron 1')
    
    axes[1,1].plot(ip_lesion[1],color = "purple",marker="o")
    axes[1,1].plot(ip_var[1],color="yellow",marker="^")
    axes[1,1].title.set_text('HL 2, Neuron 2')
    
    axes[1,2].plot(ip_lesion[2],color = "purple",marker="o")
    axes[1,2].plot(ip_var[2],color="yellow",marker="^")
    axes[1,2].title.set_text('HL 2, Neuron 3')
    
    axes[1,3].plot(ip_lesion[3],color = "purple",marker="o")
    axes[1,3].plot(ip_var[3],color="yellow",marker="^")
    axes[1,3].title.set_text('HL 2, Neuron 4')
    
    axes[1,4].plot(ip_lesion[4],color = "purple",marker="o")
    axes[1,4].plot(ip_lesion[4],color="yellow",marker="^")
    axes[1,4].title.set_text('HL 2, Neuron 5')

    #axes[2,5].title.set_text('lesions versus variance')

    figure.tight_layout()
    figure.suptitle('lesions versus variance: IP Task')
    plt.show()
    
    

