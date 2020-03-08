#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
from numpy import linalg as LA
import networkx.linalg.algebraicconnectivity as alg
import collections 
import math
import pandas as pd
import networkx as nx 


# In[2]:


def Operator(P, V, gamma, reward, epsilon):
    
    n = np.size(P, 1)
    V_1 = np.dot((1-gamma)*(np.linalg.inv(np.eye(n) - gamma*P)), reward)
    
    '''for i in range(1000):
        
        V_2 = gamma*np.dot(P,V_1) +(1-gamma)*reward
        if(np.max(np.abs(V_1 - V_2)) < epsilon):
            return V_2
        V_1 = V_2'''
    return V_1


# In[3]:


def ncut(cluster_name, w, n):
    
    cluster_name = np.asarray(cluster_name)
    cluster_name = np.reshape(cluster_name, (1,n))
    
    mull_1 = cluster_name
    mull_2 = np.ones((1,n)) - mull_1
    
    w_1 = np.dot(mull_1, np.dot(w, np.ones((n,1))))
    w_2 = np.dot(mull_2, np.dot(w, np.ones((n,1))))
    
    
    cut = np.dot(mull_1, np.dot(w,mull_2.T))
    #print(cut, w_1,w_2)
    
    ans = cut*(1/w_1 + 1/w_2)
    ans_1 = cut/min(w_1, w_2)
    
    return ans[0][0], ans_1[0][0] 


# In[4]:


def get_sweep_cut(W):
    
    gamma = 0.5
    start = 1
    
    n = np.size(W, 1)
    D = W.sum(axis = 1)
    zero_entries = np.where(D == 0)[0]
    D[zero_entries] = 1
    D_1 = np.reciprocal(D)
    D_1 = np.diag(D_1)  
    D = np.diag(D)
    P = np.dot(W, D_1)
    
    X_s = np.zeros((n,1))
    X_s[start] = 1
    V = Operator(P, np.zeros((n,1)), gamma, X_s, 0.001)
    
    order = np.argsort(np.dot(V.T, D_1), axis = 1)[0]
    
    threshold_index = 0
    best_conductance = 2
    cluster_name = [0]*n
    clusters = [[], []]

    
    for i in range(0, n - 1):

        cluster_name[order[i]] = 1
        nc, c = ncut(cluster_name, W, n)
        #print(i, nc, c, best_conductance)
        if(c <= best_conductance):
            best_conductance = c
            threshold_index = i

    cluster_name = [0]*n

    for i in range(0,threshold_index + 1):
        cluster_name[order[i]] = 1

    for i in range(0, n):
        clusters[cluster_name[i]].append(i)

    # this part changes the cluster_name from a list to numpy array (This step helps to write easy codes)
    cluster_name = np.asarray(cluster_name)
    cluster_name = np.reshape(cluster_name, (1,n))
    #print(deg)

    return best_conductance


