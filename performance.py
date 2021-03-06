

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
duration_IP = 10
stepsize_IP = 0.05
duration_CP = 50
stepsize_CP = 0.05
duration_LW = 220.0
stepsize_LW = 0.1
time_IP = np.arange(0.0,duration_IP,stepsize_IP)
time_CP = np.arange(0.0,duration_CP,stepsize_CP)
time_LW = np.arange(0.0,duration_LW,stepsize_LW)

MaxFit = 0.627 #Leggedwalker

# Fitness initialization ranges
#Inverted Pendulum
trials_theta_IP = 6
trials_thetadot_IP = 6
total_trials_IP = trials_theta_IP*trials_thetadot_IP
theta_range_IP = np.linspace(-np.pi, np.pi, num=trials_theta_IP)
thetadot_range_IP = np.linspace(-1.0,1.0, num=trials_thetadot_IP)

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

#Legged walker
trials_theta = 3
theta_range_LW = np.linspace(-np.pi/6, np.pi/6, num=trials_theta)
trials_omega_LW = 3
omega_range_LW = np.linspace(-1.0, 1.0, num=trials_omega_LW)
total_trials_LW = trials_theta * trials_omega_LW

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
    return fitness,fitness[0],fitness[1],fitness[2],fit_IP,fit_CP,fit_LW,nn_state_ip,nn_state_cp,nn_state_lw

gens = len(np.load(dir+"/average_history_15.npy"))
gs=len(np.load(dir+"/best_individual_15.npy"))
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

plt.figure(figsize=(3.2,2.7))

best_list = []
for i in range(reps):
    af[i] = np.load(dir+"/average_history_"+str(i)+".npy")
    bf[i] = np.load(dir+"/best_history_"+str(i)+".npy")
    bi[i] = np.load(dir+"/best_individual_"+str(i)+".npy")
    
    f,f1,f2,f3,m1,m2,m3,ns1,ns2,ns3=analysis(bi[i])
    np.save(dir+"/perfmap_IP_"+str(i)+".npy",m1) #behavioral analysis based on starting conditions
    np.save(dir+"/perfmap_CP_"+str(i)+".npy",m2)
    np.save(dir+"/perfmap_LW_"+str(i)+".npy",m3)

    np.save(dir+"/state_IP_"+str(i)+".npy",ns1) #to be used for infolesion analysis
    np.save(dir+"/state_CP_"+str(i)+".npy",ns2)
    np.save(dir+"/state_LW_"+str(i)+".npy",ns3)
    
# =============================================================================
#     plt.imshow(m1) #plotting fit_IP
#     plt.colorbar()
#     plt.xlabel("Theta")
#     plt.ylabel("ThetaDot")
#     plt.title("Inverted Pendulum")
#     plt.savefig(dir+"/perfmap_IP_"+str(i)+".png")
#     plt.show()
#     plt.close()
#              
#     plt.imshow(m2) #plotting fit_CP
#     plt.colorbar()
#     plt.xlabel("Theta")
#     plt.ylabel("ThetaDot")
#     plt.title("Cart Pole")
#     plt.savefig(dir+"/perfmap_CP_"+str(i)+".png")
#     plt.show()
#     plt.close()
# #             
#              plt.imshow(m3)
#              plt.colorbar()
#              plt.xlabel("Theta")
#              plt.ylabel("ThetaDot")
#              plt.title("Legged Walker")
#              plt.savefig(dir+"/perfmap_LW_"+str(i)+".png")
#              plt.show()
#              plt.close()
# =============================================================================
    
    if bf[i][-1] >= 0.8: #check last element in bf
        fit_IP_list_good.append(f1)
        fit_CP_list_good.append(f2)
        fit_LW_list_good.append(f3)
        
        
        plt.plot(bf[i].T,'b')
        #if bf[i][-1] >=0.95:
            #plt.plot(bf[i].T,'r')
        
    else:
        fit_IP_list_bad.append(f1)
        fit_CP_list_bad.append(f2)
        fit_LW_list_bad.append(f3)
        #plt.plot(af[i].T,'k')
        plt.plot(bf[i].T,'y')
    #if bf[i][-1]>=0.95:
        #plt.plot(bf[i].T,'r')
        
    best_list.append(bf[i][-1])
        
#print(best_list)
best_network = best_list.index(max(best_list))
#print(best_network)

plt.plot(bf[best_network].T,'r',label="Best Network")
plt.xlabel('generations')
plt.ylabel('performance')
plt.title('Evolution of Multifunctional Circuits')
plt.legend()
plt.tight_layout()
plt.savefig(dir+"/evolve_all_NEW.png")
plt.show()
plt.close()

#print(max([sublist for sublist in best_net_list])

# =============================================================================
# =============================================================================
#          np.save(dir+"/perfmap_IP_"+str(i)+".npy",m1) #behavioral analysis based on starting conditions
#          np.save(dir+"/perfmap_CP_"+str(i)+".npy",m2)
#          np.save(dir+"/perfmap_LW_"+str(i)+".npy",m3)
#  
#          np.save(dir+"/state_IP_"+str(i)+".npy",ns1) #to be used for infolesion analysis
#          np.save(dir+"/state_CP_"+str(i)+".npy",ns2)
#          np.save(dir+"/state_LW_"+str(i)+".npy",ns3)
# =============================================================================
# 
# =============================================================================
        #print('rep,best fitness, fitness for 3 tasks',i,bf[i][-1],f)
   
        
# =============================================================================
# if viz == 1:
#     plt.plot(af.T,'r')
#     plt.plot(bf.T,'b')
#     plt.plot(bi.T,'y')
#     plt.show()
# 
# =============================================================================

# =============================================================================
#         if viz == 1:
#             plt.imshow(m1) #plotting fit_IP
#             plt.colorbar()
#             plt.xlabel("Theta")
#             plt.ylabel("ThetaDot")
#             plt.title("Inverted Pendulum")
#             plt.savefig(dir+"/perfmap_IP_"+str(i)+".png")
#             #plt.show()
#             plt.close()
#             
#             plt.imshow(m2) #plotting fit_CP
#             plt.colorbar()
#             plt.xlabel("Theta")
#             plt.ylabel("ThetaDot")
#             plt.title("Cart Pole")
#             plt.savefig(dir+"/perfmap_CP_"+str(i)+".png")
#             #plt.show()
#             plt.close()
#             
#             plt.imshow(m3)
#             plt.colorbar()
#             plt.xlabel("Theta")
#             plt.ylabel("ThetaDot")
#             plt.title("Legged Walker")
#             plt.savefig(dir+"/perfmap_LW_"+str(i)+".png")
#             #plt.show()
#             plt.close()
# 
# =============================================================================
#print(fit_IP_list_bad)
#print(fit_CP_list_bad)
#print(fit_LW_list_bad)


plt.figure(figsize=(6.9,2.5))

plt.subplot(1,3,1)

x_cp = fit_CP_list_good
y_lw = fit_LW_list_good
x2_cp = fit_CP_list_bad
y2_lw = fit_LW_list_bad
plt.ylim(0.8,1.01)
plt.xlim(0.78,1.03)
plt.plot(x_cp, y_lw, 'bo', x2_cp, y2_lw, 'yo')
plt.plot(max(x_cp),max(y_lw),'ro', label='best overall fitness')
plt.xlabel('CP Fitness')
plt.ylabel('LW Fitness')
plt.title('(A)')

plt.subplot(1,3,2)

x_new = fit_IP_list_good
y_new = fit_LW_list_good
x2_new = fit_IP_list_bad
y2_new = fit_LW_list_bad
plt.ylim(0.8,1.01)
plt.xlim(0.78,1.03)
plt.plot(x_new, y_new, 'bo',x2_new,y2_new,'yo')
plt.plot(max(x_new),max(y_new),'ro')
plt.xlabel('IP Fitness')
plt.ylabel('LW Fitness')
plt.title('(B)')

plt.subplot(1,3,3)

x = fit_IP_list_good   
y = fit_CP_list_good
x2 = fit_IP_list_bad
y2 = fit_CP_list_bad
plt.ylim(0.8,1.01)
plt.xlim(0.78,1.03)
plt.plot(x, y, 'bo', x2, y2, 'yo')
plt.plot(max(x),max(y),'ro')
plt.xlabel('IP Fitness')
plt.ylabel('CP Fitness')
plt.title('(C)')
plt.tight_layout()
plt.savefig('scatters.png')
plt.show()
plt.close()
#     
#     

        
        
# =============================================================================
# if viz == 1:
#     plt.plot(af.T,'y')
#     plt.plot(bf.T,'b')
#     plt.xlabel("Generations")
#     plt.ylabel("Fitness")
#     plt.title("Evolution")
#     plt.savefig(dir+"/evol.png")
#     plt.show()
#     plt.close()
#     
#     x = fit_IP_list
#     y = fit_CP_list
#     plt.plot(x, y, 'o', color='black')
#     plt.xlabel('IP Fitness')
#     plt.ylabel('CP Fitness')
#     plt.title('IP vs CP Fitness')
#     plt.savefig(dir+"/IP_vs_CP_scatter"+".png")
#     plt.show()
#     plt.close()
#     
#     x = fit_CP_list
#     y = fit_LW_list
#     plt.plot(x, y, 'o', color='black')
#     plt.xlabel('CP Fitness')
#     plt.ylabel('LW Fitness')
#     plt.title('CP vs LW Fitness')
#     plt.savefig(dir+"/CP_vs_LW_scatter"+".png")
#     plt.show()
#     plt.close()
#     
#     x = fit_IP_list
#     y = fit_LW_list
#     plt.plot(x, y, 'o', color='black')
#     plt.xlabel('IP Fitness')
#     plt.ylabel('LW Fitness')
#     plt.title('IP vs LW Fitness')
#     plt.savefig(dir+"/IP_vs_LW_scatter"+".png")
#     plt.show()
#     plt.close()
# =============================================================================
    
    
    
  
# =============================================================================
# =======
# 
# import numpy as np
# import ffann                #Controller
# import invpend              #Task 1
# import cartpole             #Task 2
# import leggedwalker         #Task 3
# import matplotlib.pyplot as plt
# import sys
# 
# dir = str(sys.argv[1])
# reps = int(sys.argv[2])
# 
# # ANN Params
# nI = 3+4+3
# nH1 = 5
# nH2 = 5
# nO = 1+1+3 #output activation needs to account for 3 outputs in leggedwalker
# WeightRange = 15.0
# BiasRange = 15.0
# 
# # Task Params
# duration_IP = 10
# stepsize_IP = 0.05
# duration_CP = 50
# stepsize_CP = 0.05
# duration_LW = 220.0
# stepsize_LW = 0.1
# time_IP = np.arange(0.0,duration_IP,stepsize_IP)
# time_CP = np.arange(0.0,duration_CP,stepsize_CP)
# time_LW = np.arange(0.0,duration_LW,stepsize_LW)
# 
# MaxFit = 0.627 #Leggedwalker
# 
# # Fitness initialization ranges
# #Inverted Pendulum
# trials_theta_IP = 6
# trials_thetadot_IP = 6
# total_trials_IP = trials_theta_IP*trials_thetadot_IP
# theta_range_IP = np.linspace(-np.pi, np.pi, num=trials_theta_IP)
# thetadot_range_IP = np.linspace(-1.0,1.0, num=trials_thetadot_IP)
# 
# #Cartpole
# trials_theta_CP = 4
# trials_thetadot_CP = 4
# trials_x_CP = 4
# trials_xdot_CP = 4
# total_trials_CP = trials_theta_CP*trials_thetadot_CP*trials_x_CP*trials_xdot_CP
# theta_range_CP = np.linspace(-0.05, 0.05, num=trials_theta_CP)
# thetadot_range_CP = np.linspace(-0.05, 0.05, num=trials_thetadot_CP)
# x_range_CP = np.linspace(-0.05, 0.05, num=trials_x_CP)
# xdot_range_CP = np.linspace(-0.05, 0.05, num=trials_xdot_CP)
# 
# #Legged walker
# trials_theta = 3
# theta_range_LW = np.linspace(-np.pi/6, np.pi/6, num=trials_theta)
# trials_omega_LW = 3
# omega_range_LW = np.linspace(-1.0, 1.0, num=trials_omega_LW)
# total_trials_LW = trials_theta * trials_omega_LW
# 
# # Fitness function
# def analysis(genotype):
#     # Common setup
#     nn = ffann.ANN(nI,nH1,nH2,nO)
#     nn.setParameters(genotype,WeightRange,BiasRange)
#     fitness = np.zeros(3)
# 
#     # Task 1
#     body = invpend.InvPendulum()
#     nn_state_ip = np.zeros((total_trials_IP*len(time_IP),nI+nH1+nH2+nO))
#     total_steps = len(theta_range_IP) * len(thetadot_range_IP) * len(time_IP)
#     fit_IP = np.zeros((len(theta_range_IP),len(thetadot_range_IP)))
#     i=0
#     k=0
#     for theta in theta_range_IP:
#         #print(len(theta_range_IP))
#         j=0
#         for theta_dot in thetadot_range_IP:
#             body.theta = theta
#             body.theta_dot = theta_dot
#             f = 0.0
#             for t in time_IP:
#                 nn.step(np.concatenate((body.state(),np.zeros(4),np.zeros(3)))) #arrays for inputs for each task
#                 nn_state_ip[k] = nn.states() #re-sets the activations for every k
#                 k += 1
#                 f += body.step(stepsize_IP, np.array([nn.output()[0]]))
#             fit_IP[i][j] = ((f/duration_IP)+7.65)/7
#             j += 1
#         i += 1
#     fitness[0] = np.mean(fit_IP)
#     
#     # Task 2
#     body = cartpole.Cartpole()
#     nn_state_cp = np.zeros((total_trials_CP*len(time_CP),nI+nH1+nH2+nO))
#     #total_steps = len(theta_range_CP) * len(thetadot_range_CP) * len(x_range_CP) * len(xdot_range_CP) * len(time_CP)
#     fit_CP = np.zeros((len(theta_range_CP),len(thetadot_range_CP)))
#     i = 0
#     k = 0
#     for theta in theta_range_CP:
#         j = 0
#         for theta_dot in thetadot_range_CP:
#             f_cumulative = 0
#             for x in x_range_CP:
#                 for x_dot in xdot_range_CP:
#                     body.theta = theta
#                     body.theta_dot = theta_dot
#                     body.x = x
#                     body.x_dot = x_dot
#                     f = 0.0
#                     for t in time_CP:
#                         nn.step(np.concatenate((np.zeros(3),body.state(),np.zeros(3))))
#                         nn_state_cp[k] = nn.states()
#                         k += 1
#                         f += body.step(stepsize_CP, np.array([nn.output()[1]]))
#                     f_cumulative += f/duration_CP
#                 fit_CP[i][j] = f_cumulative/(len(x_range_CP)*len(xdot_range_CP))
#             j += 1
#         i += 1
#     fitness[1] = np.mean(fit_CP)
# 
#     #Task 3
#     body = leggedwalker.LeggedAgent(0.0,0.0)
#     nn_state_lw = np.zeros((total_trials_LW*len(time_LW),nI+nH1+nH2+nO))
#     #total_steps = len(theta_range_LW) * len(omega_range_LW) * len(time_LW)
#     fit_LW = np.zeros((len(theta_range_LW),len(omega_range_LW)))
#     i = 0
#     k = 0
#     for theta in theta_range_LW:
#         j = 0
#         for omega in omega_range_LW:
#             body.reset()
#             body.angle = theta
#             body.omega = omega
#             for t in time_LW:
#                 nn.step(np.concatenate((np.zeros(3),np.zeros(4),body.state())))
#                 nn_state_lw[k] = nn.states()
#                 k += 1
#                 body.step(stepsize_LW, np.array(nn.output()[2:5]))
#             fit_LW[i][j] = (body.cx/duration_LW)/MaxFit
#             j += 1
#         i += 1
#     fitness[2] = np.mean(fit_LW)
#     return fitness,fitness[0],fitness[1],fitness[2],fit_IP,fit_CP,fit_LW,nn_state_ip,nn_state_cp,nn_state_lw
# 
# gens = len(np.load(dir+"/average_history_15.npy"))
# gs=len(np.load(dir+"/best_individual_15.npy"))
# af = np.zeros((reps,gens))
# bf = np.zeros((reps,gens))
# bi = np.zeros((reps,gs))
# 
# fit_IP_list_good = []
# fit_CP_list_good = []
# fit_LW_list_good = []
# fit_IP_list_bad = []
# fit_CP_list_bad = []
# fit_LW_list_bad = []
# #best_net_list = []
# for i in range(reps):
#     af[i] = np.load(dir+"/average_history_"+str(i)+".npy")
#     bf[i] = np.load(dir+"/best_history_"+str(i)+".npy")
#     bi[i] = np.load(dir+"/best_individual_"+str(i)+".npy")
#     #best_net_list.append(bi[i])
#     
#     f,f1,f2,f3,m1,m2,m3,ns1,ns2,ns3=analysis(bi[i])
#     if bf[i][-1] > 0.8: #check last element in bf
#         fit_IP_list_good.append(f1)
#         fit_CP_list_good.append(f2)
#         fit_LW_list_good.append(f3)
# 
#         #plt.plot(bf[i].T,'b')
#         #if bf[i][-1] >=0.95:
#             #plt.plot(bf[i].T,'r')
#         
#     if bf[i][-1]<0.8:
#         fit_IP_list_bad.append(f1)
#         fit_CP_list_bad.append(f2)
#         fit_LW_list_bad.append(f3)
#         #plt.plot(af[i].T,'k')
#         #plt.plot(bf[i].T,'y')
#     #if bf[i][-1]>=0.95:
#         #plt.plot(bf[i].T,'r')
#         
#         
# #t = max([sublist[-1] for sublist in best_net_list])
# #for t in (sublist for sublist in best_net_list):
# #    print(t)
# #    plt.plot(sublist.T,'y')
#         
# #plt.xlabel('generations')
# #plt.ylabel('performance')
# #plt.title('evolution')
# #plt.savefig(dir+"/evolve_all.png")
# #plt.show()
# 
# #print(max([sublist for sublist in best_net_list])
# 
# # =============================================================================
# #         np.save(dir+"/perfmap_IP_"+str(i)+".npy",m1) #behavioral analysis based on starting conditions
# #         np.save(dir+"/perfmap_CP_"+str(i)+".npy",m2)
# #         np.save(dir+"/perfmap_LW_"+str(i)+".npy",m3)
# # 
# #         np.save(dir+"/state_IP_"+str(i)+".npy",ns1) #to be used for infolesion analysis
# #         np.save(dir+"/state_CP_"+str(i)+".npy",ns2)
# #         np.save(dir+"/state_LW_"+str(i)+".npy",ns3)
# # 
# # =============================================================================
#         #print('rep,best fitness, fitness for 3 tasks',i,bf[i][-1],f)
#    
#         
# # =============================================================================
# # if viz == 1:
# #     plt.plot(af.T,'r')
# #     plt.plot(bf.T,'b')
# #     plt.plot(bi.T,'y')
# #     plt.show()
# # 
# # =============================================================================
# 
# # =============================================================================
# #         if viz == 1:
# #             plt.imshow(m1) #plotting fit_IP
# #             plt.colorbar()
# #             plt.xlabel("Theta")
# #             plt.ylabel("ThetaDot")
# #             plt.title("Inverted Pendulum")
# #             plt.savefig(dir+"/perfmap_IP_"+str(i)+".png")
# #             #plt.show()
# #             plt.close()
# #             
# #             plt.imshow(m2) #plotting fit_CP
# #             plt.colorbar()
# #             plt.xlabel("Theta")
# #             plt.ylabel("ThetaDot")
# #             plt.title("Cart Pole")
# #             plt.savefig(dir+"/perfmap_CP_"+str(i)+".png")
# #             #plt.show()
# #             plt.close()
# #             
# #             plt.imshow(m3)
# #             plt.colorbar()
# #             plt.xlabel("Theta")
# #             plt.ylabel("ThetaDot")
# #             plt.title("Legged Walker")
# #             plt.savefig(dir+"/perfmap_LW_"+str(i)+".png")
# #             #plt.show()
# #             plt.close()
# # 
# # =============================================================================
#         
# x = fit_IP_list_good       
# y = fit_CP_list_good
# x2 = fit_IP_list_bad
# y2 = fit_CP_list_bad
# plt.ylim(0.8,1.01)
# plt.xlim(0.8,1.01)
# plt.plot(x, y, 'ro', x2, y2, 'bo')
# plt.xlabel('IP Fitness')
# plt.ylabel('CP Fitness')
# plt.title('IP vs CP Fitness')
# plt.savefig(dir+"/IP_vs_CP_scatter"+".png")
# plt.show()
# plt.close()
# #     
# x = fit_CP_list_good
# y = fit_LW_list_good
# x2 = fit_CP_list_bad
# y2 = fit_LW_list_bad
# plt.ylim(0.8,1.01)
# plt.xlim(0.8,1.01)
# plt.plot(x, y, 'ro',x2,y2,'bo')
# plt.xlabel('CP Fitness')
# plt.ylabel('LW Fitness')
# plt.title('CP vs LW Fitness')
# plt.savefig(dir+"/CP_vs_LW_scatter"+".png")
# plt.show()
# plt.close()
# #     
# x = fit_IP_list_good
# y = fit_LW_list_good
# x2 = fit_IP_list_bad
# y2 = fit_LW_list_bad
# plt.ylim(0.8,1.01)
# plt.xlim(0.8,1.01)
# plt.plot(x, y, 'ro',x2,y2,'bo')
# plt.xlabel('IP Fitness')
# plt.ylabel('LW Fitness')
# plt.title('IP vs LW Fitness')
# plt.savefig(dir+"/IP_vs_LW_scatter"+".png")
# plt.show()
# plt.close()
#         
#         
# # =============================================================================
# # if viz == 1:
# #     plt.plot(af.T,'y')
# #     plt.plot(bf.T,'b')
# #     plt.xlabel("Generations")
# #     plt.ylabel("Fitness")
# #     plt.title("Evolution")
# #     plt.savefig(dir+"/evol.png")
# #     plt.show()
# #     plt.close()
# #     
# #     x = fit_IP_list
# #     y = fit_CP_list
# #     plt.plot(x, y, 'o', color='black')
# #     plt.xlabel('IP Fitness')
# #     plt.ylabel('CP Fitness')
# #     plt.title('IP vs CP Fitness')
# #     plt.savefig(dir+"/IP_vs_CP_scatter"+".png")
# #     plt.show()
# #     plt.close()
# #     
# #     x = fit_CP_list
# #     y = fit_LW_list
# #     plt.plot(x, y, 'o', color='black')
# #     plt.xlabel('CP Fitness')
# #     plt.ylabel('LW Fitness')
# #     plt.title('CP vs LW Fitness')
# #     plt.savefig(dir+"/CP_vs_LW_scatter"+".png")
# #     plt.show()
# #     plt.close()
# #     
# #     x = fit_IP_list
# #     y = fit_LW_list
# #     plt.plot(x, y, 'o', color='black')
# #     plt.xlabel('IP Fitness')
# #     plt.ylabel('LW Fitness')
# #     plt.title('IP vs LW Fitness')
# #     plt.savefig(dir+"/IP_vs_LW_scatter"+".png")
# #     plt.show()
# #     plt.close()
# # =============================================================================
#     
#     
#     
#   
# >>>>>>> e8fbe70b727e80f4fd5102d1e97bc3356029035b
# =============================================================================
