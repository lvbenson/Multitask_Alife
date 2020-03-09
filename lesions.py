
import numpy as np
import ffann                #Controller
import invpend              #Task 1
import cartpole             #Task 2
import leggedwalker         #Task 3
import matplotlib.pyplot as plt
import sys

dir = str(sys.argv[1])
ind = int(sys.argv[2])
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
trials_theta_IP = 6
trials_thetadot_IP = 6
total_trials_IP = trials_theta_IP*trials_thetadot_IP
theta_range_IP = np.linspace(-np.pi, np.pi, num=trials_theta_IP)
thetadot_range_IP = np.linspace(-1.0,1.0, num=trials_thetadot_IP)

#Cartpole
trials_theta_CP = 2
trials_thetadot_CP = 2
trials_x_CP = 2
trials_xdot_CP = 2
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

def lesions(genotype,actvalues):

    nn = ffann.ANN(nI,nH1,nH2,nO)

    # Task 1
    ip_fit = np.zeros(nH1+nH2)
    body = invpend.InvPendulum()
    nn.setParameters(genotype,WeightRange,BiasRange)
    index = 0
    for layer in [1,2]:
        for neuron in range(nH1):
            if layer == 1:
                n = neuron
            else:
                n = nH1 + neuron
            #print("IP:",n)
            maxfit = 0.0
            for act in actvalues[:,0,n]:
                fit = 0.0
                for theta in theta_range_IP:
                    for theta_dot in thetadot_range_IP:
                        body.theta = theta
                        body.theta_dot = theta_dot
                        for t in time_IP:
                            nn.step_lesioned(np.concatenate((body.state(),np.zeros(4),np.zeros(3))),neuron,layer,act)
                            f = body.step(stepsize_IP, np.array([nn.output()[0]]))
                            fit += f
                fit = fit/(duration_IP*total_trials_IP)
                fit = (fit+7.65)/7
                if fit < 0.0:
                    fit = 0.0
                if fit < maxfit:
                    maxfit = fit
            ip_fit[index]=fit
            index += 1

    # Task 2
    cp_fit = np.zeros(nH1+nH2)
    body = cartpole.Cartpole()
    nn.setParameters(genotype,WeightRange,BiasRange)
    index = 0
    for layer in [1,2]:
        for neuron in range(nH1):
            if layer == 1:
                n = neuron
            else:
                n = nH1 + neuron
            #print("CP:",n)
            maxfit = 0.0
            for act in actvalues[:,1,n]:
                fit = 0.0
                for theta in theta_range_CP:
                    for theta_dot in thetadot_range_CP:
                        for x in x_range_CP:
                            for x_dot in xdot_range_CP:
                                body.theta = theta
                                body.theta_dot = theta_dot
                                body.x = x
                                body.x_dot = x_dot
                                for t in time_CP:
                                    nn.step_lesioned(np.concatenate((np.zeros(3),body.state(),np.zeros(3))),neuron,layer,act)
                                    f = body.step(stepsize_CP, np.array([nn.output()[1]]))
                                    fit += f
                fit = fit/(duration_CP*total_trials_CP)
                if fit < 0.0:
                    fit = 0.0
                if fit < maxfit:
                    maxfit = fit
            cp_fit[index]=fit
            index += 1

    #Task 3
    lw_fit = np.zeros(nH1+nH2)
    body = leggedwalker.LeggedAgent(0.0,0.0)
    nn.setParameters(genotype,WeightRange,BiasRange)
    index = 0
    for layer in [1,2]:
        for neuron in range(nH1):
            if layer == 1:
                n = neuron
            else:
                n = nH1 + neuron
            #print("LW:",n)
            maxfit = 0.0
            for act in actvalues[:,2,n]:
                fit = 0.0
                for theta in theta_range_LW:
                    for omega in omega_range_LW:
                        body.reset()
                        body.angle = theta
                        body.omega = omega
                        for t in time_LW:
                            nn.step_lesioned(np.concatenate((np.zeros(3),np.zeros(4),body.state())),neuron,layer,act)
                            body.step(stepsize_LW, np.array(nn.output()[2:5]))
                        fit += body.cx/duration_LW
                fit = (fit/total_trials_LW)/MaxFit
                if fit < 0.0:
                    fit = 0.0
                if fit < maxfit:
                    maxfit = fit
            lw_fit[index]=fit
            index += 1

    return ip_fit,cp_fit,lw_fit

max = np.zeros((3,nH1+nH2))
nn = np.load("./{}/state_IP_{}.npy".format(dir,ind))
max[0] = np.max(nn[:,nI:nI+nH1+nH2],axis=0)
nn = np.load("./{}/state_CP_{}.npy".format(dir,ind))
max[1] = np.max(nn[:,nI:nI+nH1+nH2],axis=0)
nn = np.load("./{}/state_LW_{}.npy".format(dir,ind))
max[2] = np.max(nn[:,nI:nI+nH1+nH2],axis=0)

steps = 10
actvalues = np.linspace(0.0, max, num=steps)

bi = np.load("./{}/best_individual_{}.npy".format(dir,ind))
f = np.load("./{}/perf_{}.npy".format(dir,ind))

ipp,cpp,lwp = lesions(bi,actvalues)

ipp = ipp/f[0]
cpp = cpp/f[1]
lwp = lwp/f[2]

np.save(dir+"/lesions_IP_"+str(ind)+".npy",ipp)
np.save(dir+"/lesions_CP_"+str(ind)+".npy",cpp)
np.save(dir+"/lesions_LW_"+str(ind)+".npy",lwp)

if viz == 1:
    plt.plot(ipp,'ro')
    plt.plot(cpp,'go')
    plt.plot(lwp,'bo')
    plt.xlabel("Interneuron")
    plt.ylabel("Performance")
    plt.title("Lesions")
    plt.savefig(dir+"/lesions_"+str(ind)+".png")
    plt.show()

# Stats on neurons for Ablations
Threshold = 0.95
count = np.zeros(8)
for (ip_neuron, cp_neuron, lw_neuron) in zip(ipp,cpp,lwp):
    if ip_neuron > Threshold and cp_neuron > Threshold and lw_neuron > Threshold: # no task neurons
        count[0] += 1
    if ip_neuron <= Threshold and cp_neuron > Threshold and lw_neuron > Threshold: # ip task neurons
        count[1] += 1
    if ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron > Threshold: # cp task neurons
        count[2] += 1
    if ip_neuron > Threshold and cp_neuron > Threshold and lw_neuron <= Threshold: #lw task neurons
        count[3] += 1
    if ip_neuron <= Threshold and cp_neuron <= Threshold and lw_neuron > Threshold: # ip + cp task neurons
        count[4] += 1
    if ip_neuron <= Threshold and cp_neuron > Threshold and lw_neuron <= Threshold: # ip + lw task neurons
        count[5] += 1
    if ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # cp + lw task neurons
        count[6] += 1
    if ip_neuron <=  Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # all  task neurons
        count[7] += 1

np.save(dir+"/stats_"+str(ind)+".npy",count)
#print(count)
