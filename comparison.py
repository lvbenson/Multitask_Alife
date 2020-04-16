# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:29:15 2020

@author: Lauren Benson
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import math

dir = str(sys.argv[1])
reps = int(sys.argv[2])

for ind in range(reps):

    bf = np.load(dir+"/best_history_"+str(ind)+".npy")
    
    if bf[-1]>0.8:

        # load lesions data
        ip_lesion = np.load("./{}/lesions_IP_{}.npy".format(dir,ind))
        cp_lesion = np.load("./{}/lesions_CP_{}.npy".format(dir,ind))
        lw_lesion = np.load("./{}/lesions_LW_{}.npy".format(dir,ind))
    
        lesion = np.concatenate((ip_lesion,cp_lesion,lw_lesion),axis=None)

        # load variance data
        ip_var = 1-np.load("./{}/NormVar_IP_{}.npy".format(dir,ind))
        cp_var = 1-np.load("./{}/NormVar_CP_{}.npy".format(dir,ind))
        lw_var = 1-np.load("./{}/NormVar_LW_{}.npy".format(dir,ind))

        var = np.concatenate((ip_var,cp_var,lw_var),axis=None)

        # load mutual information data
        ip_mi = 1-np.load("./{}/NormMI_IP_{}.npy".format(dir,ind))
        cp_mi = 1-np.load("./{}/NormMI_CP_{}.npy".format(dir,ind))
        lw_mi = 1-np.load("./{}/NormMI_LW_{}.npy".format(dir,ind))

        mi = np.concatenate((ip_mi,cp_mi,lw_mi),axis=None)

        # distance between vectors
        
        
        dLV = math.sqrt(sum((lesion-var)**2))
        dLM = math.sqrt(sum((lesion-mi)**2))
        dVM = math.sqrt(sum((var-mi)**2))
        
        colors = ['b', 'g', 'r']
        Lesion_Var = plt.scatter(dLV, np.zeros_like(dLV), marker='o', color=colors[0])
        Lesion_MI = plt.scatter(dLM, np.zeros_like(dLM), marker='o', color=colors[1])
        Var_MI = plt.scatter(dVM, np.zeros_like(dLM), marker='o', color=colors[2])
        
        plt.legend((Lesion_Var,Lesion_MI,Var_MI),
           ('Dist: Lesion & Var','Dist: Lesion & MI','Dist: Var & MI'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
        plt.yticks([])
        plt.xlabel('Euclidean Distance')
        plt.title('Lesion & Var, Lesion & MI, Var & MI')
        plt.savefig(dir+"/Euclid_Distance_"+str(ind)+".png")
        plt.show()
        plt.close()


        # lesions matrix
        new_ip_lesion = np.reshape(ip_lesion,(5,2))
        new_cp_lesion = np.reshape(cp_lesion,(5,2))
        new_lw_lesion = np.reshape(lw_lesion,(5,2))

        # variance matrix
        new_ip_var = np.reshape(ip_var,(5,2))
        new_cp_var = np.reshape(cp_var,(5,2))
        new_lw_var = np.reshape(lw_var,(5,2))

        # mutual information matrix
        new_ip_mi = np.reshape(ip_mi,(5,2))
        new_cp_mi = np.reshape(cp_mi,(5,2))
        new_lw_mi = np.reshape(lw_mi,(5,2))

        # visualize lesion
        stack = 1-np.dstack((new_ip_lesion,new_cp_lesion,new_lw_lesion))
        plt.imshow(stack)
        plt.xlabel('layers')
        plt.ylabel('interneurons')
        plt.title('Lesions')
        plt.savefig(dir+"/Viz_Lesion_"+str(ind)+".png")
        plt.show()
        plt.close()

        # visualize variance
        var_stack = 1-np.dstack((new_ip_var,new_cp_var,new_lw_var))
        plt.imshow(var_stack)
        plt.xlabel('Layers')
        plt.ylabel('interneurons')
        plt.title('Variance')
        plt.savefig(dir+"/Viz_Var_"+str(ind)+".png")
        plt.show()
        plt.close()

        # visualize mutual info
        mi_stack = 1-np.dstack((new_ip_mi,new_cp_mi,new_lw_mi))
        plt.imshow(mi_stack)
        plt.xlabel('layers')
        plt.ylabel('interneurons')
        plt.title('Mutual Information')
        plt.savefig(dir+"/Viz_MI_"+str(ind)+".png")
        plt.show()
        plt.close()

        # Plot performance vs neurons for all three methods. Compare.
        
        # plot lesions
        plt.plot(ip_lesion, color="red",marker = "o", linestyle = 'none')
        plt.plot(cp_lesion,color="green",marker = "o",linestyle = 'none')
        plt.plot(lw_lesion,color="blue",marker = "o",linestyle = 'none')
        plt.xlabel("Interneuron")
        plt.ylabel("Performance")
        plt.title("Lesions")
        plt.savefig(dir+"/Marker_LesionsX_"+str(ind)+".png")
        plt.show()
        plt.close()

        # plot variance
        plt.plot(ip_var, color="red",marker = "^",linestyle = 'none')
        plt.plot(cp_var, color="green",marker = "^",linestyle = 'none')
        plt.plot(lw_var, color="blue",marker = "^",linestyle = 'none')
        plt.xlabel("Interneuron")
        plt.ylabel("Performance")
        plt.title("Variance")
        plt.savefig(dir+"/Marker_VarX_"+str(ind)+".png")
        plt.show()
        plt.close()

        # plot mutual information
        plt.plot(ip_mi, color="red",marker = "s", linestyle = 'none')
        plt.plot(cp_mi, color="green",marker = "s", linestyle = 'none')
        plt.plot(lw_mi, color="blue",marker = "s", linestyle = 'none')
        plt.xlabel("Interneuron")
        plt.ylabel("Performance")
        plt.title("MutualInfo")
        plt.savefig(dir+"/Marker_MIX_"+str(ind)+".png")
        plt.show()
        plt.close()
