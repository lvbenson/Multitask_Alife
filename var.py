import numpy as np
import matplotlib.pyplot as plt
import sys

dir = str(sys.argv[1])
ind = int(sys.argv[2])

def find_all_var(dir,ind):
#variance measures how far a set of numbers are spread out from their average value
    
    nI = 10
    nH = 10
    v = np.zeros((3,10))
    nn = np.load("./{}/state_IP_{}.npy".format(dir,ind))
    #
    v[0] = np.var(nn[:,nI:nI+nH],axis=0)
    nn = np.load("./{}/state_CP_{}.npy".format(dir,ind))
    v[1] = np.var(nn[:,nI:nI+nH],axis=0)
    nn = np.load("./{}/state_LW_{}.npy".format(dir,ind))
    v[2] = np.var(nn[:,nI:nI+nH],axis=0)
    v = v.T
    max = np.max(v,axis=1)
    norm_var = (v.T / max)
    plt.plot(norm_var[0],'ro') #ip
    np.save(dir+"/Var_IP_"+str(ind)+".npy",norm_var[0])
    #save IP Variance
    plt.plot(norm_var[1],'go') #cp
    np.save(dir+"/Var_CP_"+str(ind)+".npy",norm_var[1])
    #save CP Variance
    plt.plot(norm_var[2],'bo') #lw
    np.save(dir+"/Var_LW_"+str(ind)+".npy",norm_var[2])
    #save LW Variance
    plt.xlabel("Interneurons")
    plt.ylabel("Normalized variance")
    plt.title("Normalized variance")
    plt.savefig(dir+"/var_"+str(ind)+".png")
    plt.show()

find_all_var(dir,ind)
