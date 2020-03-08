#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

from RAH import show_results_hypothesis
from RAH import check_partion
from ClusteringUsingValue import get_sweep_cut
from ClusteringPagerank import get_sweep_cut_pagerank
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
    Results_ncut = pd.DataFrame(index = ['SLS','SLP', 'SLH', 'UnS','UnP', 'UnH','RwS','RwP', 'RwH'])
    Results_conductance = pd.DataFrame(index = ['SLS','SLP', 'SLH', 'UnS','UnP', 'UnH','RwS','RwP', 'RwH'])

    Result_V = []
    Result_Pr = []
    for i in range(0,100):
        
        generate_random_graph(n, p, file_name)
        data = np.load(file_name)
        M = data['arr_0']
        
        if check_partion(M, n):
            continue
        
        Result_V.append(get_sweep_cut(M))
        Result_Pr.append(get_sweep_cut_pagerank(M))
        Graph = ['g' + str(i), M]
        Results_ncut[Graph[0]], Results_conductance[Graph[0]] = show_results_hypothesis(Graph[1])

    RL = Results_conductance.T

    y_SLS = np.sum(RL['SLS'].values)
    y_SLP = np.sum(RL['SLP'].values)
    y_SLH = np.sum(RL['SLH'].values)
    y_UnS = np.sum(RL['UnS'].values)
    y_UnP = np.sum(RL['UnP'].values)
    y_UnH = np.sum(RL['UnH'].values)
    y_RwS = np.sum(RL['RwS'].values)
    y_RwP = np.sum(RL['RwP'].values)
    y_RwH = np.sum(RL['RwH'].values)
    y_V = sum(Result_V)
    y_Pr = sum(Result_Pr)
    
    return y_SLS, y_SLP, y_SLH, y_UnS, y_UnP, y_UnH, y_RwS, y_RwP, y_RwH, y_V, y_Pr

