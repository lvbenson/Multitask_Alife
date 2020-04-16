<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 09:40:08 2020

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
#print(ip_lesion[0])
cp_lesion = np.load("./{}/lesions_CP_{}.npy".format(dir,ind))
lw_lesion = np.load("./{}/lesions_LW_{}.npy".format(dir,ind))

#load variance data
ip_var = np.load("./{}/NormVar_IP_{}.npy".format(dir,ind))
cp_var = np.load("./{}/NormVar_CP_{}.npy".format(dir,ind))
lw_var = np.load("./{}/NormVar_LW_{}.npy".format(dir,ind))

ip_mi = np.load("./{}/NormMI_IP_{}.npy".format(dir,ind))
cp_mi = np.load("./{}/NormMI_CP_{}.npy".format(dir,ind))
lw_mi = np.load("./{}/NormMI_LW_{}.npy".format(dir,ind))

#lesions matrix

new_ip_lesion = np.reshape(ip_lesion,(5,2))
new_cp_lesion = np.reshape(cp_lesion,(5,2))
new_lw_lesion = np.reshape(lw_lesion,(5,2))

#print(new_ip_lesion.shape)

#variance matrix

new_ip_var = np.reshape(ip_var,(5,2))
new_cp_var = np.reshape(cp_var,(5,2))
new_lw_var = np.reshape(lw_var,(5,2))

#mutual information matrix

new_ip_mi = np.reshape(ip_mi,(5,2))
new_cp_mi = np.reshape(cp_mi,(5,2))
new_lw_mi = np.reshape(lw_mi,(5,2))

#create (5,2,3) shape for valid RGB input
stack = np.dstack((new_ip_lesion,new_cp_lesion,new_lw_lesion))

#print(stack.shape)

plt.imshow(stack)
plt.xlabel('layers')
plt.ylabel('interneurons')
plt.title('Lesions')
plt.savefig(dir+"/Real_Lesions_"+str(ind)+".png")
#plt.colorbar()
plt.show()
plt.close()

var_stack = np.dstack((new_ip_var,new_cp_var,new_lw_var))
plt.imshow(var_stack)
plt.xlabel('Layers')
plt.ylabel('interneurons')
plt.title('Variance')
plt.savefig(dir+"/Real_Var_"+str(ind)+".png")
#plt.colorbar()
plt.show()
plt.close()

mi_stack = np.dstack((new_ip_mi,new_cp_mi,new_lw_mi))
plt.imshow(mi_stack)
plt.xlabel('layers')
plt.ylabel('interneurons')
plt.title('Mutual Information')
plt.savefig(dir+"/Real_MI_"+str(ind)+".png")
#plt.colorbar()
plt.show()
plt.close()

#categories matrix

#lesions





    

=======
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 09:40:08 2020

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
#print(ip_lesion[0])
cp_lesion = np.load("./{}/lesions_CP_{}.npy".format(dir,ind))
lw_lesion = np.load("./{}/lesions_LW_{}.npy".format(dir,ind))

#load variance data
ip_var = np.load("./{}/NormVar_IP_{}.npy".format(dir,ind))
cp_var = np.load("./{}/NormVar_CP_{}.npy".format(dir,ind))
lw_var = np.load("./{}/NormVar_LW_{}.npy".format(dir,ind))

ip_mi = np.load("./{}/NormMI_IP_{}.npy".format(dir,ind))
cp_mi = np.load("./{}/NormMI_CP_{}.npy".format(dir,ind))
lw_mi = np.load("./{}/NormMI_LW_{}.npy".format(dir,ind))

#lesions matrix

new_ip_lesion = np.reshape(ip_lesion,(5,2))
new_cp_lesion = np.reshape(cp_lesion,(5,2))
new_lw_lesion = np.reshape(lw_lesion,(5,2))

#print(new_ip_lesion.shape)

#variance matrix

new_ip_var = np.reshape(ip_var,(5,2))
new_cp_var = np.reshape(cp_var,(5,2))
new_lw_var = np.reshape(lw_var,(5,2))

#mutual information matrix

new_ip_mi = np.reshape(ip_mi,(5,2))
new_cp_mi = np.reshape(cp_mi,(5,2))
new_lw_mi = np.reshape(lw_mi,(5,2))

#create (5,2,3) shape for valid RGB input
stack = np.dstack((new_ip_lesion,new_cp_lesion,new_lw_lesion))

#print(stack.shape)

plt.imshow(stack)
plt.xlabel('layers')
plt.ylabel('interneurons')
plt.title('Lesions')
plt.savefig(dir+"/Real_Lesions_"+str(ind)+".png")
#plt.colorbar()
plt.show()
plt.close()

var_stack = np.dstack((new_ip_var,new_cp_var,new_lw_var))
plt.imshow(var_stack)
plt.xlabel('Layers')
plt.ylabel('interneurons')
plt.title('Variance')
plt.savefig(dir+"/Real_Var_"+str(ind)+".png")
#plt.colorbar()
plt.show()
plt.close()

mi_stack = np.dstack((new_ip_mi,new_cp_mi,new_lw_mi))
plt.imshow(mi_stack)
plt.xlabel('layers')
plt.ylabel('interneurons')
plt.title('Mutual Information')
plt.savefig(dir+"/Real_MI_"+str(ind)+".png")
#plt.colorbar()
plt.show()
plt.close()

#categories matrix

#lesions





    

>>>>>>> e8fbe70b727e80f4fd5102d1e97bc3356029035b
