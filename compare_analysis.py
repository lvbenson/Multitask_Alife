# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 09:40:08 2020

@author: Lauren Benson
"""

import numpy as np
import matplotlib.pyplot as plt
import sys


#v iz to compare: (1) variance against lesions. (2) mutual information against lesions.
# Lesions = "ground truth"

dir = str(sys.argv[1])
ind = int(sys.argv[2])
viz = int(sys.argv[3])

#load lesions data
ip_lesion = np.load("./{}/Lesions_IP_{}.npy".format(dir,ind))
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
    plt.title("Variance: Predictive Measure")
    plt.savefig(dir+"/Compare_"+str(ind)+".png")
    plt.show()

#Finer-grain comparison
#variance vs infolesion
    

    

