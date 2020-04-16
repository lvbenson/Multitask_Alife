# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 12:40:25 2020

@author: Lauren Benson
"""

#need combined histograms on 1 axes
#first for all the networks with fitness above 0.9, 
#then for all the ones with fitness above 0.8. 



import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from matplotlib import colors


dir = str(sys.argv[1])
reps = int(sys.argv[2])

dLV_list = []
dLM_list = []
dVM_list = []

count = 0

for ind in range(reps):

    bf = np.load(dir+"/best_history_"+str(ind)+".npy")
    
    
    if bf[-1]>0.8:
        
        count += 1

        # load lesions data
        ip_lesion = np.load("./{}/lesions_IP_{}.npy".format(dir,ind))
        cp_lesion = np.load("./{}/lesions_CP_{}.npy".format(dir,ind))
        lw_lesion = np.load("./{}/lesions_LW_{}.npy".format(dir,ind))
    
        lesion = np.concatenate((ip_lesion,cp_lesion,lw_lesion),axis=None)
        #size 30; ip, cp, lw for the 10 interneurons 

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
        
        #calculate distances (3 values)
        dist_list = []
        dLV = math.sqrt(sum((lesion-var)**2))
        dLM = math.sqrt(sum((lesion-mi)**2))
        dVM = math.sqrt(sum((var-mi)**2))
        
        #to use for combined histogram
        dist_list.append(dLV)
        dist_list.append(dLM)
        dist_list.append(dVM)
        
        #to use for separate histograms
        dLV_list.append(dLV)
        dLM_list.append(dLM)
        dVM_list.append(dVM)
        
        x = dLM_list
        y = dLV_list
        plt.plot(x, y, 'o', color='black')
        for i, txt in enumerate(x):
        plt.annotate(txt, (x[i], y[i]))
        plt.xlabel('Lesions-Mutual Info')
        plt.ylabel('Lesions-Variance')
        plt.title('Euclidean Distance')
        plt.savefig(dir+"/Euclid_Distance_Hist_scatter"+".png")
        plt.show()





print(count)
        
        
#side by side histograms
# =============================================================================
# 
# fig, axs = plt.subplots(1, 3, sharey=True, constrained_layout=True)
# N, bins, patches = axs[0].hist(dLV_list, bins='auto')
# fracs = N / N.max()
# norm = colors.Normalize(fracs.min(), fracs.max())
# for thisfrac, thispatch in zip(fracs, patches):
#     color = plt.cm.viridis(norm(thisfrac))
#     thispatch.set_facecolor(color)
# 
# N, bins, patches = axs[1].hist(dLM_list, bins='auto')
# fracs = N / N.max()
# norm = colors.Normalize(fracs.min(), fracs.max())
# for thisfrac, thispatch in zip(fracs, patches):
#     color = plt.cm.viridis(norm(thisfrac))
#     thispatch.set_facecolor(color)
# 
# N, bins, patches = axs[2].hist(dVM_list, bins='auto')
# fracs = N / N.max()
# norm = colors.Normalize(fracs.min(), fracs.max())
# for thisfrac, thispatch in zip(fracs, patches):
#     color = plt.cm.viridis(norm(thisfrac))
#     thispatch.set_facecolor(color)
# 
# axs[0].set_title('Lesions,Var',fontsize=10)
# axs[0].set_xlabel('Distance')
# axs[1].set_title('Lesions,MI',fontsize=10)
# axs[1].set_xlabel('Distance')
# axs[2].set_title('Var,MI',fontsize=10)
# axs[2].set_xlabel('Distance')
# fig.suptitle('Euclidean Distance, fitness>0.8',fontsize=20)
# plt.savefig(dir+"/Euclid_Distance_Hist0.8"+".png")
# plt.show()
# plt.close()
# =============================================================================


#combined histograms

# =============================================================================
# _, bins, _ = plt.hist(dLM_list, bins=40, range=[0, 5], density=True, label='Lesions,MI')
# _ = plt.hist(dVM_list, bins=bins, alpha=0.5, density=True, label='Variance,MI')
# _ = plt.hist(dLV_list, bins=bins, alpha=0.5, density=True, label='Lesions,Variance')
# plt.ylabel('Distance')
# plt.title('Euclidean Distance, fitness>0.8')
# plt.legend()
# plt.savefig(dir+"/Euclid_Distance_Hist_combined0.8"+".png")
# plt.show()
# plt.close()
# =============================================================================

_, bins, _ = plt.hist(dLM_list, bins=30, range=[0, 5], density=True, label='Lesions,MI')
_ = plt.hist(dLV_list, bins=bins, alpha=0.5, density=True, label='Lesions,Variance')
plt.ylabel('Distance')
plt.title('Euclidean Distance, fitness>0.8')
plt.legend()
plt.savefig(dir+"/Euclid_Distance_Hist_combined0.8_2"+".png")
plt.show()
plt.close()

x = dLM_list
y = dLV_list
plt.plot(x, y, 'o', color='black')
for i, txt in enumerate(x):
    plt.annotate(txt, (x[i], y[i]))
plt.xlabel('Lesions-Mutual Info')
plt.ylabel('Lesions-Variance')
plt.title('Euclidean Distance')
plt.savefig(dir+"/Euclid_Distance_Hist_scatter"+".png")
plt.show()


#now for all fitness > 0.9
# =============================================================================
# 
# dLV_list_2 = []
# dLM_list_2 = []
# dVM_list_2 = []
# 
# newcount = 0
# 
# for ind in range(reps):
# 
#     bf = np.load(dir+"/best_history_"+str(ind)+".npy")
#     
#     
#     if bf[-1]>0.9:
#         
#         newcount += 1
#         
#         # load lesions data
#         ip_lesion = np.load("./{}/lesions_IP_{}.npy".format(dir,ind))
#         cp_lesion = np.load("./{}/lesions_CP_{}.npy".format(dir,ind))
#         lw_lesion = np.load("./{}/lesions_LW_{}.npy".format(dir,ind))
#     
#         lesion = np.concatenate((ip_lesion,cp_lesion,lw_lesion),axis=None)
#         #size 30; ip, cp, lw for the 10 interneurons 
# 
#         # load variance data
#         ip_var = 1-np.load("./{}/NormVar_IP_{}.npy".format(dir,ind))
#         cp_var = 1-np.load("./{}/NormVar_CP_{}.npy".format(dir,ind))
#         lw_var = 1-np.load("./{}/NormVar_LW_{}.npy".format(dir,ind))
# 
#         var = np.concatenate((ip_var,cp_var,lw_var),axis=None)
# 
#         # load mutual information data
#         ip_mi = 1-np.load("./{}/NormMI_IP_{}.npy".format(dir,ind))
#         cp_mi = 1-np.load("./{}/NormMI_CP_{}.npy".format(dir,ind))
#         lw_mi = 1-np.load("./{}/NormMI_LW_{}.npy".format(dir,ind))
# 
#         mi = np.concatenate((ip_mi,cp_mi,lw_mi),axis=None)
#         
#         #calculate distances (3 values)
#         dLV = math.sqrt(sum((lesion-var)**2))
#         dLM = math.sqrt(sum((lesion-mi)**2))
#         dVM = math.sqrt(sum((var-mi)**2))
#         
#         
#         #to use for separate histograms
#         dLV_list_2.append(dLV)
#         dLM_list_2.append(dLM)
#         dVM_list_2.append(dVM)
#         
# 
# print(newcount)
# #side by side histograms
# 
# fig, axs = plt.subplots(1, 3, sharey=True, constrained_layout=True)
# N, bins, patches = axs[0].hist(dLV_list_2, bins='auto')
# fracs = N / N.max()
# norm = colors.Normalize(fracs.min(), fracs.max())
# for thisfrac, thispatch in zip(fracs, patches):
#     color = plt.cm.viridis(norm(thisfrac))
#     thispatch.set_facecolor(color)
# 
# N, bins, patches = axs[1].hist(dLM_list_2, bins='auto')
# fracs = N / N.max()
# norm = colors.Normalize(fracs.min(), fracs.max())
# for thisfrac, thispatch in zip(fracs, patches):
#     color = plt.cm.viridis(norm(thisfrac))
#     thispatch.set_facecolor(color)
# 
# N, bins, patches = axs[2].hist(dVM_list_2, bins='auto')
# fracs = N / N.max()
# norm = colors.Normalize(fracs.min(), fracs.max())
# for thisfrac, thispatch in zip(fracs, patches):
#     color = plt.cm.viridis(norm(thisfrac))
#     thispatch.set_facecolor(color)
# 
# axs[0].set_title('Lesions,Var',fontsize=10)
# axs[0].set_xlabel('Distance')
# axs[1].set_title('Lesions,MI',fontsize=10)
# axs[1].set_xlabel('Distance')
# axs[2].set_title('Var,MI',fontsize=10)
# axs[2].set_xlabel('Distance')
# fig.suptitle('Euclidean Distance, fitness > 0.9',fontsize=20)
# plt.savefig(dir+"/Euclid_Distance_Hist0.9"+".png")
# plt.show()
# plt.close()
# 
# 
# #combined histograms
# 
# _, bins, _ = plt.hist(dLM_list_2, bins=30, range=[0, 5], density=True, label='Lesions,MI')
# _ = plt.hist(dVM_list_2, bins=bins, alpha=0.5, density=True, label='Variance,MI')
# _ = plt.hist(dLV_list_2, bins=bins, alpha=0.5, density=True, label='Lesions,Variance')
# plt.ylabel('Distance')
# plt.title('Euclidean Distance, fitness>0.9')
# plt.legend()
# plt.savefig(dir+"/Euclid_Distance_Hist_combined0.9"+".png")
# plt.show()
# 
# _, bins, _ = plt.hist(dLM_list_2, bins=30, range=[0, 5], density=True, label='Lesions,MI')
# _ = plt.hist(dLV_list_2, bins=bins, alpha=0.5, density=True, label='Lesions,Variance')
# plt.ylabel('Distance')
# plt.title('Euclidean Distance, fitness>0.9')
# plt.legend()
# plt.savefig(dir+"/Euclid_Distance_Hist_combined0.9_2"+".png")
# plt.show()
# 
# 
# =============================================================================
