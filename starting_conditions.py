
import numpy as np
import ffann                #Controller
import invpend              #Task 1
import cartpole             #Task 2
import leggedwalker         #Task 3
import matplotlib.pyplot as plt
import sys

dir = str(sys.argv[1])
reps = int(sys.argv[2])
viz = int(sys.argv[3])

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

# Fitness function
def analysis(genotype):
    # Common setup
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    
    
    #task 1
    newfitness_IP = np.zeros(20)
    
    for r in range(1,20,1):
        
        trials_theta_IP = r
        trials_thetadot_IP = r
        total_trials_IP = trials_theta_IP*trials_thetadot_IP
        theta_range_IP = np.linspace(-np.pi, np.pi, num=trials_theta_IP)
        thetadot_range_IP = np.linspace(-1.0,1.0, num=trials_thetadot_IP)

        body = invpend.InvPendulum()
        nn_state_ip = np.zeros((total_trials_IP*len(time_IP),nI+nH1+nH2+nO))
        fit_IP = np.zeros((len(theta_range_IP),len(thetadot_range_IP)))
        i=0
        k=0
        for theta in theta_range_IP:
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
        newfitness_IP[r] = np.mean(fit_IP)
    #print(newfitness_IP)
    print(len(newfitness_IP))
    #task 2
     
   
    newfitness_CP = np.zeros(10)
    
    for condition in range(1,10,1):
        trials_theta_CP = condition
        trials_thetadot_CP = condition
        trials_x_CP = condition
        trials_xdot_CP = condition
        total_trials_CP = trials_theta_CP*trials_thetadot_CP*trials_x_CP*trials_xdot_CP
        theta_range_CP = np.linspace(-0.05, 0.05, num=trials_theta_CP)
        thetadot_range_CP = np.linspace(-0.05, 0.05, num=trials_thetadot_CP)
        x_range_CP = np.linspace(-0.05, 0.05, num=trials_x_CP)
        xdot_range_CP = np.linspace(-0.05, 0.05, num=trials_xdot_CP)
        
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
        newfitness_CP[condition] = np.mean(fit_CP)
        #print('condition',newfitness_CP[condition])
    #print('cp fitness',newfitness_CP)
    print(len(newfitness_CP))
        
        
   
    return newfitness_IP,newfitness_CP


gens = len(np.load(dir+"/average_history_0.npy"))
gs=len(np.load(dir+"/best_individual_0.npy"))
af = np.zeros((reps,gens))
bf = np.zeros((reps,gens))
bi = np.zeros((reps,gs))
for i in range(reps):
    af[i] = np.load(dir+"/average_history_"+str(i)+".npy")
    bf[i] = np.load(dir+"/best_history_"+str(i)+".npy")
    bi[i] = np.load(dir+"/best_individual_"+str(i)+".npy")
    if bf[i][-1]>0.8:

        f1,f2=analysis(bi[i])
        
        plt.plot(f1, linestyle='--', marker='o', color='b')
        plt.xlabel("Starting Condition")
        plt.ylabel("Performance")
        plt.title("Performance over starting conditions: CP")
        plt.show()
        plt.close()
        
        plt.plot(f2, linestyle='--', marker='o', color='b')
        plt.xlabel("Starting Condition")
        plt.ylabel("Performance")
        plt.title("Performance over starting conditions: CP")
        plt.show()
        plt.close()
        
# =============================================================================
#         plt.plot(f2, linestyle='--', marker='o', color='b')
#         plt.xlabel("Starting Condition")
#         plt.ylabel("Performance")
#         plt.title("Performance over starting conditions: CP")
#         plt.show()
#         plt.close()
#         
# =============================================================================
        


