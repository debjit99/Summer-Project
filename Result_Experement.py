#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import import_ipynb
from RA import show_results
from RA import check_partion

import pandas as pd
from numpy import *
import numpy as np 
import networkx as nx 


# In[2]:


def generate_random_graph(n, p, file_name):
    
    G= nx.fast_gnp_random_graph(n, p, seed=1, directed=False)
    A = nx.to_numpy_matrix(G)
    np.savez(file_name, A)
    return 


# In[3]:


def Result(n, p):
    
    path = os.getcwd()
    file_name = path + '/' + 'dummy.npz'
    Results_ncut = pd.DataFrame(index = ['SLUP','SLP', 'SLS', 'UnUP','UnP', 'UnS','RwUP','RwP', 'RwS'])
    Results_conductance = pd.DataFrame(index = ['SLUP','SLP', 'SLS', 'UnUP','UnP', 'UnS','RwUP','RwP', 'RwS'])

    for i in range(0,100):
        
        generate_random_graph(n, p, file_name)
        data = np.load(file_name)
        M = data['arr_0']
        
        if check_partion(M, n):
            continue
        
        Graph = ['g' + str(i), M]
        Results_ncut[Graph[0]], Results_conductance[Graph[0]] = show_results(Graph[1])

    RL = Results_conductance.T

    y_SLP = np.sum(RL['SLP'].values)
    y_UnP = np.sum(RL['UnP'].values)
    y_RwP = np.sum(RL['RwP'].values)
    y_SLS = np.sum(RL['SLS'].values)
    y_UnS = np.sum(RL['UnS'].values)
    y_RwS = np.sum(RL['RwS'].values)
    
    return y_SLP, y_UnP, y_RwP, y_SLS, y_UnS, y_RwS

