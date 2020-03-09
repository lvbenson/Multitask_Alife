import numpy as np
import infotheory
import matplotlib.pyplot as plt
import sys

dir = str(sys.argv[1])
ind = int(sys.argv[2])

def mi(filename, nbins=50, nreps=3):

    dat = np.load(filename)
    print(dat.shape)
    nI = 10
    nH = 10

    # to iterate through all neurons
    neuron_inds = np.arange(nI,nI+nH)

    # range for sensory input
    mins = np.min(dat[:,:nI], 0)
    maxs = np.max(dat[:,:nI], 0)

    # range for neurons - min and max across neurons
    n_min = np.min(dat[:,nI:nI+nH])
    n_max = np.max(dat[:,nI:nI+nH])

    # concat ranges
    mins = np.append(mins, n_min)
    maxs = np.append(maxs, n_max)
    dims = len(mins)

    # setup for infotheory analysis
    mis = []
    var_ids = [0]*nI + [1]

    # estimate mi
    for i,neuron_ind in enumerate(neuron_inds):
        print("\tNeuron # {}".format(i+1))
        it = infotheory.InfoTools(dims, nreps)
        it.set_equal_interval_binning([nbins]*dims, mins, maxs)
        it.add_data(np.hstack([dat[:,:nI], np.array([dat[:, neuron_ind]]).T]))

        mi = it.mutual_info(var_ids)
        mis.append(mi)

    return mis

def find_all_mis(dir,ind):
    ip_mis = mi("./{}/state_IP_{}.npy".format(dir,ind))
    np.save("./{}/MI_IP_{}.npy".format(dir,ind), ip_mis)
    cp_mis = mi("./{}/state_CP_{}.npy".format(dir,ind))
    np.save("./{}/MI_CP_{}.npy".format(dir,ind), cp_mis)
    ip_mis = mi("./{}/state_LW_{}.npy".format(dir,ind))
    np.save("./{}/MI_LW_{}.npy".format(dir,ind), ip_mis)

def plot_mis(dir,ind):
    ip_mi = np.load("./{}/mi_IP_{}.npy".format(dir,ind))
    cp_mi = np.load("./{}/mi_CP_{}.npy".format(dir,ind))
    lw_mi = np.load("./{}/mi_LW_{}.npy".format(dir,ind))
    plt.plot(ip_mi,'ro')
    plt.plot(cp_mi,'go')
    plt.plot(lw_mi,'bo')
    plt.xlabel("Interneurons")
    plt.ylabel("Mutual information")
    plt.title("Mutual Information")
    plt.savefig(dir+"/mi_"+str(ind)+".png")
    plt.show()

find_all_mis(dir,ind)
plot_mis(dir,ind)
