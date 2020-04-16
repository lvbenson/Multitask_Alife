# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:15:50 2020

@author: Lauren Benson
"""

import numpy as np
import matplotlib.pyplot as plt
import sys


#viz to compare: (1) variance against lesions. (2) mutual information against lesions.
# Lesions = "ground truth"
#use directory: Local3


dir = str(sys.argv[1])
ind = int(sys.argv[2])

#load lesions data
ip_lesion = np.load("./{}/lesions_IP_{}.npy".format(dir,ind))
cp_lesion = np.load("./{}/lesions_CP_{}.npy".format(dir,ind))
lw_lesion = np.load("./{}/lesions_LW_{}.npy".format(dir,ind))

#load variance data
ip_var = 1-np.load("./{}/NormVar_IP_{}.npy".format(dir,ind))
#print(ip_var)
cp_var = 1-np.load("./{}/NormVar_CP_{}.npy".format(dir,ind))
#print(cp_var)
lw_var = 1-np.load("./{}/NormVar_LW_{}.npy".format(dir,ind))
#print(lw_var)

ip_mi = 1-np.load("./{}/NormMI_IP_{}.npy".format(dir,ind))
cp_mi = 1-np.load("./{}/NormMI_CP_{}.npy".format(dir,ind))
lw_mi = 1-np.load("./{}/NormMI_LW_{}.npy".format(dir,ind))

real_ip_lesion = np.reshape(ip_lesion,(2,5))
real_cp_lesion = np.reshape(cp_lesion,(2,5))
real_lw_lesion = np.reshape(lw_lesion,(2,5))

#variance matrix

real_ip_var = np.reshape(ip_var,(2,5))
real_cp_var = np.reshape(cp_var,(2,5))
real_lw_var = np.reshape(lw_var,(2,5))

#mutual information matrix

real_ip_mi = np.reshape(ip_mi,(2,5))
real_cp_mi = np.reshape(cp_mi,(2,5))
real_lw_mi = np.reshape(lw_mi,(2,5))

#create (5,2,3) shape for valid RGB input
lesion_stack = 1-np.dstack((real_ip_lesion,real_cp_lesion,real_lw_lesion))
plt.imshow(lesion_stack)
plt.xlabel('Interneurons')
plt.yticks([])
plt.ylabel('Layers')
plt.title('Real data: Lesions')
plt.savefig(dir+"/Real_LesionsX_"+str(ind)+".png")
plt.show()
plt.close()

var_stack = 1-np.dstack((real_ip_var,real_cp_var,real_lw_var))
plt.imshow(var_stack)
plt.xlabel('Interneurons')
plt.yticks([])
plt.ylabel('Layers')
plt.title('Real data: Variance')
plt.savefig(dir+"/Real_VarX_"+str(ind)+".png")
#plt.colorbar()
plt.show()
plt.close()

mi_stack = 1-np.dstack((real_ip_mi,real_cp_mi,real_lw_mi))
plt.imshow(mi_stack)
plt.xlabel('Interneurons')
plt.yticks([])
plt.ylabel('Layers')
plt.title('Real data: Mutual Information')
plt.savefig(dir+"/Real_MIX_"+str(ind)+".png")
plt.show()
plt.close()


############
#Plot lesions
############


plt.plot(ip_lesion, color="red",marker = "o", linestyle = 'none')
plt.plot(cp_lesion,color="green",marker = "o",linestyle = 'none')
plt.plot(lw_lesion,color="blue",marker = "o",linestyle = 'none')
plt.axhline(y=0.99, color='c', linestyle='-')
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
plt.axhline(y=0.99, color='c', linestyle='-')
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
plt.axhline(y=0.99, color='c', linestyle='-')
plt.xlabel("Interneuron")
plt.ylabel("Performance")
plt.title("MutualInfo")
plt.savefig(dir+"/Marker_MIX_"+str(ind)+".png")
plt.show()
plt.close()







#0 in lesion = neuron is very involved, 1 means neuron is not involved

#CATEGORIES ANALYSIS

Threshold = 0.99

for n, ip_neuron in enumerate(ip_lesion):
    if ip_neuron > Threshold:
        ip_lesion[n] = 1
    else:
        ip_lesion[n] = 0
        
for n, cp_neuron in enumerate(cp_lesion):
    if cp_neuron > Threshold:
        cp_lesion[n] = 1
    else:
        cp_lesion[n] = 0
        
for n, lw_neuron in enumerate(lw_lesion):
    if lw_neuron > Threshold:
        lw_lesion[n] = 1
    else:
        lw_lesion[n] = 0
    


cat_ip_lesion = np.reshape(ip_lesion,(2,5))
cat_cp_lesion = np.reshape(cp_lesion,(2,5))
cat_lw_lesion = np.reshape(lw_lesion,(2,5))    

#dstack concatenates along third dimension
cat_stack_lesion = 1-np.dstack((cat_ip_lesion,cat_cp_lesion,cat_lw_lesion))

#variance categories
#0 means neuron is not very involved

Threshold = 0.99

for n, ip_neuron in enumerate(ip_var):
    if ip_neuron > Threshold:
        ip_var[n] = 1
    else:
        ip_var[n] = 0
        
for n, cp_neuron in enumerate(cp_var):
    if cp_neuron > Threshold:
        cp_var[n] = 1
    else:
        cp_var[n] = 0
        
for n, lw_neuron in enumerate(lw_var):
    if lw_neuron > Threshold:
        lw_var[n] = 1
    else:
        lw_var[n] = 0

cat_ip_var = np.reshape(ip_var,(2,5))
cat_cp_var = np.reshape(cp_var,(2,5))
cat_lw_var = np.reshape(lw_var,(2,5))    

cat_stack_var = 1-np.dstack((cat_ip_var,cat_cp_var,cat_lw_var))


#mutual information categories

Threshold = 0.99

for n, ip_neuron in enumerate(ip_mi):
    if ip_neuron > Threshold:
        ip_mi[n] = 1
    else:
        ip_mi[n] = 0
        
for n, cp_neuron in enumerate(cp_mi):
    if cp_neuron > Threshold:
        cp_mi[n] = 1
    else:
        cp_mi[n] = 0
        
for n, lw_neuron in enumerate(lw_mi):
    if lw_neuron > Threshold:
        lw_mi[n] = 1
    else:
        lw_mi[n] = 0

cat_ip_mi = np.reshape(ip_mi,(2,5))
cat_cp_mi = np.reshape(cp_mi,(2,5))
cat_lw_mi = np.reshape(lw_mi,(2,5))    

cat_stack_mi = 1-np.dstack((cat_ip_mi,cat_cp_mi,cat_lw_mi))



plt.imshow(cat_stack_lesion)
plt.xlabel('Interneurons')
plt.yticks([])
plt.ylabel('Layers')
plt.title('Lesions: Categories')
plt.savefig(dir+"/Cat_LesionX_"+str(ind)+".png")
plt.show()
plt.close()

plt.imshow(cat_stack_var)
plt.xlabel('Interneurons')
plt.yticks([])
plt.ylabel('Layers')
plt.title('Variance: Categories')
plt.savefig(dir+"/Cat_VarX_"+str(ind)+".png")
plt.show()
plt.close()

plt.imshow(cat_stack_mi)
plt.xlabel('Interneurons')
plt.yticks([])
plt.ylabel('Layers')
plt.title('Mutual Information: Categories')
plt.savefig(dir+"/Cat_MIX_"+str(ind)+".png")
plt.show()
plt.close()



################
#Euclidean Distance scatter plot
##############################







