####
# generic plotter for two swarm plots
# have to configure npy file names and indices of data to plot
####

import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import seaborn as sns

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
            if ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # cp + lw task neuron
                cp_lw_task_list.append(1)
            if ip_neuron <=  Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # all  task neurons
                all_task_list.append(1)
                    
    d = [[len(no_task_list)],[len(ip_task_list)],[len(cp_task_list)],[len(lw_task_list)],[len(ip_cp_task_list)],[len(ip_lw_task_list)],[len(cp_lw_task_list)],[len(all_task_list)]]
    #print(d)
    num_plots = len(d)
    ax = plt.gca()
    dat = np.concatenate(d)
    data_inds = np.concatenate([[i] * len(d[i]) for i in range(num_plots)])
    data = np.vstack([data_inds, dat]).T
    data = pd.DataFrame(data=data, columns=["type", "info"])
    sns.stripplot("type","info", data=data, jitter=0.4, size=3.5)
    #sns.swarmplot(
        #"type", "info", data=data, order=np.arange(num_plots), ax=ax,
        #)
    sns.despine()
        
    plt.xlabel("Categories")
    plt.ylabel("# of neurons in each category")
    plt.xticks(np.arange(num_plots),('No tasks','IP','CP','LW','IP&CP','IP&LW','CP&LW','All Tasks'))
    plt.title('Lesions, Swarms')
    #plt.ylabel(ylabel)
    ax = plt.gca()
    # Hide the right and top spines
    #ax.spines["right"].set_visible(False)
    #ax.spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    plt.tight_layout()
plt.show()
plt.close()





# =============================================================================
# 
# 
# 
# for ind in range(reps):
#   bf = np.load(dir+"/best_history_"+str(ind)+".npy")
#   if bf[-1]>0.8:
#     ip_var = np.load("./{}/NormVar_IP_{}.npy".format(dir,ind))
#     cp_var = np.load("./{}/NormVar_CP_{}.npy".format(dir,ind))
#     lw_var = np.load("./{}/NormVar_LW_{}.npy".format(dir,ind))
#     Threshold = 0.95
#     no_task_var = []
#     ip_task_var = []
#     cp_task_var = []
#     lw_task_var = []
#     ip_cp_task_var = []
#     ip_lw_task_var = []
#     cp_lw_task_var = []
#     all_task_var = []
#     
#     for (ip_neuron, cp_neuron, lw_neuron) in zip(ip_var,cp_var,lw_var):
#       if ip_neuron > Threshold and cp_neuron > Threshold and lw_neuron > Threshold: # no task neurons
#         no_task_var.append(ip_neuron)
#         no_task_var.append(cp_neuron)
#         no_task_var.append(lw_neuron)     
#       if ip_neuron <= Threshold and cp_neuron > Threshold and lw_neuron > Threshold: # ip task neurons
#         ip_task_var.append(ip_neuron)
#       if ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron > Threshold: # cp task neurons
#         cp_task_var.append(cp_neuron)
#       if ip_neuron > Threshold and cp_neuron > Threshold and lw_neuron <= Threshold: #lw task neurons
#         lw_task_var.append(lw_neuron)
#       if ip_neuron <= Threshold and cp_neuron <= Threshold and lw_neuron > Threshold: # ip + cp task neurons
#         ip_cp_task_var.append(ip_neuron)
#         ip_cp_task_var.append(cp_neuron)
#       if ip_neuron <= Threshold and cp_neuron > Threshold and lw_neuron <= Threshold: # ip + lw task neurons
#         ip_lw_task_var.append(ip_neuron)
#         ip_lw_task_var.append(lw_neuron)
#       if ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # cp + lw task neurons
#         cp_lw_task_var.append(cp_neuron)
#         cp_lw_task_var.append(lw_neuron)
#       if ip_neuron <=  Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # all  task neurons
#         all_task_var.append(ip_neuron)
#         all_task_var.append(cp_neuron)
#         all_task_var.append(lw_neuron)
#       
#     
#     d = [no_task_var,ip_task_var,cp_task_var,lw_task_var,ip_cp_task_var,ip_lw_task_var,cp_lw_task_var,all_task_var]
#     num_plots = len(d)
#     # fig = plt.figure(figsize=[3*num_plots, 3.5])
#     ax = plt.gca()
#     dat = np.concatenate(d)
#     data_inds = np.concatenate([[i] * len(d[i]) for i in range(num_plots)])
#     data = np.vstack([data_inds, dat]).T
#     data = pd.DataFrame(data=data, columns=["type", "info"])
#     #print(data)
#     sns.swarmplot(
#         "type", "info", data=data, order=np.arange(num_plots), ax=ax,
#         )
#     plt.xlabel("Categories")
#     plt.ylabel("fitness")
#     plt.xticks(np.arange(num_plots),('No tasks','IP','CP','LW','IP&CP','IP&LW','CP&LW','All Tasks'))
#     plt.title('Variance, Swarms')
#     #plt.ylabel(ylabel)
#     ax = plt.gca()
#     # Hide the right and top spines
#     ax.spines["right"].set_visible(False)
#     ax.spines["top"].set_visible(False)
# 
#     # Only show ticks on the left and bottom spines
#     ax.yaxis.set_ticks_position("left")
#     ax.xaxis.set_ticks_position("bottom")
#     plt.tight_layout()
# 
# plt.savefig(dir+"/swarm_plot_variance_"+str(ind)+".png")
# #print(len(d))
# plt.show()
# plt.close()
# 
# 
# 
# for ind in range(reps):
#   bf = np.load(dir+"/best_history_"+str(ind)+".npy")
#   if bf[-1]>0.8:
#     ip_mi = np.load("./{}/NormMI_IP_{}.npy".format(dir,ind))
#     cp_mi = np.load("./{}/NormMI_CP_{}.npy".format(dir,ind))
#     lw_mi = np.load("./{}/NormMI_LW_{}.npy".format(dir,ind))
#     Threshold = 0.95
#     no_task_MI = []
#     ip_task_MI = []
#     cp_task_MI = []
#     lw_task_MI = []
#     ip_cp_task_MI = []
#     ip_lw_task_MI = []
#     cp_lw_task_MI = []
#     all_task_MI = []
#     
#     for (ip_neuron, cp_neuron, lw_neuron) in zip(ip_mi,cp_mi,lw_mi):
#       if ip_neuron > Threshold and cp_neuron > Threshold and lw_neuron > Threshold: # no task neurons
#         no_task_MI.append(ip_neuron)
#         no_task_MI.append(cp_neuron)
#         no_task_MI.append(lw_neuron)     
#       if ip_neuron <= Threshold and cp_neuron > Threshold and lw_neuron > Threshold: # ip task neurons
#         ip_task_MI.append(ip_neuron)
#       if ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron > Threshold: # cp task neurons
#         cp_task_MI.append(cp_neuron)
#       if ip_neuron > Threshold and cp_neuron > Threshold and lw_neuron <= Threshold: #lw task neurons
#         lw_task_MI.append(lw_neuron)
#       if ip_neuron <= Threshold and cp_neuron <= Threshold and lw_neuron > Threshold: # ip + cp task neurons
#         ip_cp_task_MI.append(ip_neuron)
#         ip_cp_task_MI.append(cp_neuron)
#       if ip_neuron <= Threshold and cp_neuron > Threshold and lw_neuron <= Threshold: # ip + lw task neurons
#         ip_lw_task_MI.append(ip_neuron)
#         ip_lw_task_MI.append(lw_neuron)
#       if ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # cp + lw task neurons
#         cp_lw_task_MI.append(cp_neuron)
#         cp_lw_task_MI.append(lw_neuron)
#       if ip_neuron <=  Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # all  task neurons
#         all_task_MI.append(ip_neuron)
#         all_task_MI.append(cp_neuron)
#         all_task_MI.append(lw_neuron)
#       
# 
#     d = [no_task_MI,ip_task_MI,cp_task_MI,lw_task_MI,ip_cp_task_MI,ip_lw_task_MI,cp_lw_task_MI,all_task_MI]
#     # fig = plt.figure(figsize=[3*num_plots, 3.5])
#     ax = plt.gca()
#     dat = np.concatenate(d)
#     data_inds = np.concatenate([[i] * len(d[i]) for i in range(num_plots)])
#     data = np.vstack([data_inds, dat]).T
#     data = pd.DataFrame(data=data, columns=["type", "info"])
#     sns.swarmplot(
#         "type", "info", data=data, order=np.arange(num_plots), ax=ax,
#         )
#     plt.xlabel("Categories")
#     plt.ylabel("fitness")
#     plt.xticks(np.arange(num_plots),('No tasks','IP','CP','LW','IP&CP','IP&LW','CP&LW','All Tasks'))
#     plt.title('Mutual Information, Swarms')
#     #plt.ylabel(ylabel)
#     ax = plt.gca()
#     # Hide the right and top spines
#     ax.spines["right"].set_visible(False)
#     ax.spines["top"].set_visible(False)
# 
#     # Only show ticks on the left and bottom spines
#     ax.yaxis.set_ticks_position("left")
#     ax.xaxis.set_ticks_position("bottom")
#     plt.tight_layout()
# 
# plt.savefig(dir+"/swarm_plot_mi_"+str(ind)+".png")
# plt.show()
# 
# 
# =============================================================================
