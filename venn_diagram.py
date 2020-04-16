<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:51:32 2020

@author: benso
"""


import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib_venn import venn3, venn3_circles


dir = str(sys.argv[1])
reps = int(sys.argv[2])

l_1 = []
l_2 = []
l_3 = []
l_4 = []
l_5 = []
l_6 = []
l_7 = []
l_8 = []

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
            if ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # cp + lw task neurons
                cp_lw_task_list.append(1)
            if ip_neuron <=  Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # all  task neurons
                all_task_list.append(1)
        #print('first',all_task_list)
        #print('cp',cp_task_list)
        l_1.append(no_task_list)
        l_2.append(ip_task_list)
        l_3.append(cp_task_list)
        l_4.append(lw_task_list)
        l_5.append(ip_cp_task_list)
        l_6.append(ip_lw_task_list)
        l_7.append(cp_lw_task_list)
        l_8.append(all_task_list)
#print(l_3)
#print('second',l_8)
#print(len(l_8))
        
count1 = 0
for listElem in l_1:
    count1 += len(listElem)    
    
#print('num neurons',count1)
#print('average',count1/len(l_1))

count2 = 0
for listElem in l_2:
    count2 += len(listElem)   

#print('num neurons2',count2)
#print('avg 2',count2/len(l_2))
      
count3 = 0
for listElem in l_3:
    count3 += len(listElem)   
    
#print('num neurons3',count3)
#print('avg 3',count3/len(l_3))    


count4 = 0
for listElem in l_4:
    count4 += len(listElem)   
count5 = 0
for listElem in l_5:
    count5 += len(listElem)  
count6 = 0
for listElem in l_6:
    count6 += len(listElem)              
count7 = 0
for listElem in l_7:
    count7 += len(listElem)   
count8 = 0
for listElem in l_8:
    count8 += len(listElem)   
 
   
#actual number of neurons, over the ensemble, for every category
v=venn3(subsets = (count2/len(l_2), count3/len(l_3), count5/len(l_5), count4/len(l_4), count6/len(l_6), count7/len(l_7), count8/len(l_8)), set_labels = ('IP Task', 'CP Task', 'LW Task'))
plt.show()
plt.close()




=======
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:51:32 2020

@author: benso
"""


import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib_venn import venn3, venn3_circles


dir = str(sys.argv[1])
reps = int(sys.argv[2])

l_1 = []
l_2 = []
l_3 = []
l_4 = []
l_5 = []
l_6 = []
l_7 = []
l_8 = []

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
            if ip_neuron > Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # cp + lw task neurons
                cp_lw_task_list.append(1)
            if ip_neuron <=  Threshold and cp_neuron <= Threshold and lw_neuron <= Threshold: # all  task neurons
                all_task_list.append(1)
        #print('first',all_task_list)
        #print('cp',cp_task_list)
        l_1.append(no_task_list)
        l_2.append(ip_task_list)
        l_3.append(cp_task_list)
        l_4.append(lw_task_list)
        l_5.append(ip_cp_task_list)
        l_6.append(ip_lw_task_list)
        l_7.append(cp_lw_task_list)
        l_8.append(all_task_list)
#print(l_3)
#print('second',l_8)
#print(len(l_8))
        
count1 = 0
for listElem in l_1:
    count1 += len(listElem)    
    
#print('num neurons',count1)
#print('average',count1/len(l_1))

count2 = 0
for listElem in l_2:
    count2 += len(listElem)   

#print('num neurons2',count2)
#print('avg 2',count2/len(l_2))
      
count3 = 0
for listElem in l_3:
    count3 += len(listElem)   
    
#print('num neurons3',count3)
#print('avg 3',count3/len(l_3))    


count4 = 0
for listElem in l_4:
    count4 += len(listElem)   
count5 = 0
for listElem in l_5:
    count5 += len(listElem)  
count6 = 0
for listElem in l_6:
    count6 += len(listElem)              
count7 = 0
for listElem in l_7:
    count7 += len(listElem)   
count8 = 0
for listElem in l_8:
    count8 += len(listElem)   
 
   
#actual number of neurons, over the ensemble, for every category
v=venn3(subsets = (count2/len(l_2), count3/len(l_3), count5/len(l_5), count4/len(l_4), count6/len(l_6), count7/len(l_7), count8/len(l_8)), set_labels = ('IP Task', 'CP Task', 'LW Task'))
plt.show()
plt.close()




>>>>>>> e8fbe70b727e80f4fd5102d1e97bc3356029035b
