import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import seaborn as sns


dir = str(sys.argv[1])
reps = int(sys.argv[2])
data = []
categories = 8
catmat = np.zeros((32,8))

##########LESIONS##########################

k=0
for ind in range(reps):
    bf = np.load(dir+"/best_history_"+str(ind)+".npy")
    if bf[-1]>0.80:
        ip_lesion = np.load("./{}/lesions_IP_{}.npy".format(dir,ind))
        cp_lesion = np.load("./{}/lesions_CP_{}.npy".format(dir,ind))
        lw_lesion = np.load("./{}/lesions_LW_{}.npy".format(dir,ind))
        Threshold = 0.99
        cat = np.zeros(categories)
        for (ip_neuron, cp_neuron, lw_neuron) in zip(ip_lesion,cp_lesion,lw_lesion):
            if ip_neuron > Threshold and cp_neuron > Threshold and lw_neuron > Threshold: # no task neurons
                cat[0] += 1
            if ip_neuron <= Threshold and cp_neuron > Threshold and lw_neuron > Threshold: # ip task neurons
                cat[1] += 1
            if ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron > Threshold: # cp task neurons
                cat[2] += 1
            if ip_neuron > Threshold and cp_neuron > Threshold and lw_neuron <= Threshold: #lw task neurons
                cat[3] += 1
            if ip_neuron <= Threshold and cp_neuron <= Threshold and lw_neuron > Threshold: # ip + cp task neurons
                cat[4] += 1
            if ip_neuron <= Threshold and cp_neuron > Threshold and lw_neuron <= Threshold: # ip + lw task neurons
                cat[5] += 1
            if ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # cp + lw task neuron
                cat[6] += 1
            if ip_neuron <=  Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # all  task neurons
                cat[7] += 1
        catmat[k]=cat
        k+=1
        for c in range(categories):
            data.append([ind,c,cat[c]])
cat_med = np.median(catmat,axis=0)
cat_1 = cat_med[0]
cat_2 = cat_med[1]
cat_3 = cat_med[2]
cat_4 = cat_med[3]
cat_5 = cat_med[4]
cat_6 = cat_med[5]
cat_7 = cat_med[6]
cat_8 = cat_med[7]



datanumpy = np.array(data)
dataframe = pd.DataFrame({'id': datanumpy[:, 0], 'Reuse Category': datanumpy[:, 1], 'num': datanumpy[:, 2]})
g = sns.catplot(x="Reuse Category", y="num", hue="id", kind="swarm", data=dataframe, aspect=1.61803398875);
g._legend.remove()
plt.ylabel("Number of neurons per neural network")
plt.xlabel("Neural Reuse Category")
plt.xticks(np.arange(categories),('None','IP','CP','LW','IP+CP','IP+LW','CP+LW','All'))
plt.tight_layout()
plt.savefig('SWARMPLOT_MI.png')
plt.show()


################## VARIANCE ##########################

datav = []
categoriesv = 8
catmatv = np.zeros((32,8))

k=0
for ind in range(reps):
    bf = np.load(dir+"/best_history_"+str(ind)+".npy")
    if bf[-1]>0.80:
        ip_var = np.load("./{}/NormVar_IP_{}.npy".format(dir,ind))
        cp_var = np.load("./{}/NormVar_CP_{}.npy".format(dir,ind))
        lw_var = np.load("./{}/NormVar_LW_{}.npy".format(dir,ind))
        Threshold = 0.99
        catv = np.zeros(categoriesv)
        for (ip_neuron, cp_neuron, lw_neuron) in zip(ip_var,cp_var,lw_var):
            if ip_neuron > Threshold and cp_neuron > Threshold and lw_neuron > Threshold: # no task neurons
                catv[0] += 1
            if ip_neuron <= Threshold and cp_neuron > Threshold and lw_neuron > Threshold: # ip task neurons
                catv[1] += 1
            if ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron > Threshold: # cp task neurons
                catv[2] += 1
            if ip_neuron > Threshold and cp_neuron > Threshold and lw_neuron <= Threshold: #lw task neurons
                catv[3] += 1
            if ip_neuron <= Threshold and cp_neuron <= Threshold and lw_neuron > Threshold: # ip + cp task neurons
                catv[4] += 1
            if ip_neuron <= Threshold and cp_neuron > Threshold and lw_neuron <= Threshold: # ip + lw task neurons
                catv[5] += 1
            if ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # cp + lw task neuron
                catv[6] += 1
            if ip_neuron <=  Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # all  task neurons
                catv[7] += 1
        catmatv[k]=catv
        k+=1
        for c in range(categoriesv):
            datav.append([ind,c,catv[c]])
cat_medv = np.median(catmatv,axis=0)
catv_1 = cat_medv[0]
catv_2 = cat_medv[1]
catv_3 = cat_medv[2]
catv_4 = cat_medv[3]
catv_5 = cat_medv[4]
catv_6 = cat_medv[5]
catv_7 = cat_medv[6]
catv_8 = cat_medv[7]




datanumpyv = np.array(datav)
dataframev = pd.DataFrame({'id': datanumpyv[:, 0], 'cat': datanumpyv[:, 1], 'num': datanumpyv[:, 2]})
gv = sns.catplot(x="cat", y="num", hue="id", kind="swarm",data=dataframev, aspect=1.61803398875);
gv._legend.remove()
#plt.ylabel("Number of neurons per neural network")
plt.xlabel("")
plt.ylabel("")
plt.xticks(np.arange(categories),('None','IP','CP','LW','IP+CP','IP+LW','CP+LW','All'))
plt.tight_layout()
plt.savefig('SWARMPLOT_MI.png')
plt.show()

################ MI ###########################

datam = []
categoriesm = 8
catmatm = np.zeros((32,8))




k=0
for ind in range(reps):
    bf = np.load(dir+"/best_history_"+str(ind)+".npy")
    if bf[-1]>0.80:
        ip_mi = np.load("./{}/NormMI_IP_{}.npy".format(dir,ind))
        cp_mi = np.load("./{}/NormMI_CP_{}.npy".format(dir,ind))
        lw_mi = np.load("./{}/NormMI_LW_{}.npy".format(dir,ind))
        Threshold = 0.99
        cati = np.zeros(categoriesm)
        for (ip_neuron, cp_neuron, lw_neuron) in zip(ip_mi,cp_mi,lw_mi):
            if ip_neuron > Threshold and cp_neuron > Threshold and lw_neuron > Threshold: # no task neurons
                cati[0] += 1
            if ip_neuron <= Threshold and cp_neuron > Threshold and lw_neuron > Threshold: # ip task neurons
                cati[1] += 1
            if ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron > Threshold: # cp task neurons
                cati[2] += 1
            if ip_neuron > Threshold and cp_neuron > Threshold and lw_neuron <= Threshold: #lw task neurons
                cati[3] += 1
            if ip_neuron <= Threshold and cp_neuron <= Threshold and lw_neuron > Threshold: # ip + cp task neurons
                cati[4] += 1
            if ip_neuron <= Threshold and cp_neuron > Threshold and lw_neuron <= Threshold: # ip + lw task neurons
                cati[5] += 1
            if ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # cp + lw task neuron
                cati[6] += 1
            if ip_neuron <=  Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # all  task neurons
                cati[7] += 1
        catmatm[k]=cati
        k+=1
        for c in range(categoriesm):
            datam.append([ind,c,cati[c]])
cat_medi = np.median(catmatm,axis=0)
cati_1 = cat_medi[0]
cati_2 = cat_medi[1]
cati_3 = cat_medi[2]
cati_4 = cat_medi[3]
cati_5 = cat_medi[4]
cati_6 = cat_medi[5]
cati_7 = cat_medi[6]
cati_8 = cat_medi[7]


datanumpym = np.array(datam)
dataframem = pd.DataFrame({'id': datanumpym[:, 0], 'cat': datanumpym[:, 1], 'num': datanumpym[:, 2]})
gm = sns.catplot(x="cat", y="num", hue="id", kind="swarm",data=dataframem, aspect=1.61803398875);
gm._legend.remove()
#plt.ylabel("Number of neurons per neural network")
plt.xlabel("")
plt.ylabel("")
plt.xticks(np.arange(categories),('None','IP','CP','LW','IP+CP','IP+LW','CP+LW','All'))

plt.tight_layout()
plt.savefig('SWARMPLOT_MI.png')
plt.show()
