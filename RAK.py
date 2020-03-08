#!/usr/bin/env python
# coding: utf-8

# This code has all algorithm which we proposed and we check it using two types of laplacian that is symmetic and unnormalized laplacian. 
# And we have set the precision for the edge weigths here so we get desired results for the Petersen graph.
# Here we have used edge weigths as (f_i - f_j)^2. 
# 
# And we are doing it for just for graphs with edge weigth 1 or 0.
# 
# # here i have added the code for random walk laplacian 
# # Changed the order in hypothesis as the fielder eigenvector order so that we get kannan algorithm only relevant entries are
# 

# In[1]:


import numpy as np
import scipy as sp
from numpy import linalg as LA
import networkx.linalg.algebraicconnectivity as alg
import collections 
import math
import pandas as pd


Results_ncut = pd.DataFrame(index = ['UnL' , 'SymL', 'RwL'])
Results_conductance = pd.DataFrame(index = ['UnL' , 'SymL', 'RwL'])
Graphs = []


# In[2]:


# This wil get us the similarity matrix W given the graph and the number of vertices n.
# the vertices are number form 0 to n - 1.
def get_matrix2(graph, n):
    
    w = np.zeros((n, n))
   
    for x in graph.keys() :
        for y in graph[x]:
            w[x][y] = 1
            w[y][x] = 1

    return w
    


# In[3]:


def get_unnormalized_laplacian(w):
    
    D = w.sum(axis = 1)
    
    D = np.diag(D)  # The degree matrix D
    
    L = D - w
    
    return L


# In[4]:


def get_symmetric_laplacian(w):
    
    D = w.sum(axis = 1)
    D_sqrt = np.sqrt(D)
    
    D_1 = np.reciprocal(D_sqrt)
    D_1 = np.diag(D_1)  
    D = np.diag(D)
    L = np.dot(D_1, np.dot(D - w, D_1))
    
    return L


# In[5]:


def get_random_walk_laplacian(w):
    
    D = w.sum(axis = 1)
    
    D_1 = np.reciprocal(D)
    D_1 = np.diag(D_1)  
    D = np.diag(D)
    L = np.dot(D_1, np.dot(D - w, D_1))
    
    return L


# In[6]:


# returns the fielder eigen vector given the similartiy matrix W.

def get_fielder(l, n):  #Similairty matix W

    eigvals, eigvecs = sp.linalg.eigh(l)# eigen values and eigen vectors of the unnormalized laplacian D - W 
    #print("Eigenvalues: ",eigvals, "\n")
    eigvecs = np.round( eigvecs, 5)
    eigvals = np.round(eigvals, 4)
    cardinality = 1
    
    fielder_eigval = eigvals[1]
    
    #print(fielder_eigval)
    while(cardinality < n and abs(eigvals[cardinality] - fielder_eigval)<0.000001):
        cardinality = cardinality + 1
        
    fielder = eigvecs.T[1, :] #The 2nd smallest eigen vector of the laplacian L = D - W
    
    return fielder


# In[7]:


def ncut(cluster_name, w, n):
    
    cluster_name = np.asarray(cluster_name)
    cluster_name = np.reshape(cluster_name, (1,n))
    
    mull_1 = cluster_name
    mull_2 = np.ones((1,n)) - mull_1
    
    w_1 = np.dot(mull_1, np.dot(w, np.ones((n,1))))
    w_2 = np.dot(mull_2, np.dot(w, np.ones((n,1))))
    
    
    cut = np.dot(mull_1, np.dot(w,mull_2.T))
    #print(cut, w_1,w_2)
    if(cut == 0):
        return 0, 0
    
    ans = cut*(1/w_1 + 1/w_2)
    ans_1 = cut/min(w_1, w_2)
    
    return ans[0][0], ans_1[0][0] 

    


# In[8]:


def get_cut(clusters, w, n):
    cut = []
    for x in clusters[0]:
        for y in clusters[1]:
            if(w[x][y] != 0):
                cut.append([x,y])
    return cut


# In[9]:


# Here we use our current algorithm to partition the graph
# This returns the clusters, the cluster_name vector and the edeges cut
def get_clusters(f, w, n): # the similarity matrix and the number of vertices
    
    threshold_index = 0
    index = np.argsort(f)
    #print(index)
    
    best_conductance = 2
    cluster_name = [0]*n
    clusters = [[], []]
    
    for i in range(0, n - 1):
        
        cluster_name[index[i]] = 1
        nc, c = ncut(cluster_name, w, n)
        #print(i, nc, c, best_conductance)
        if(c <= best_conductance):
            best_conductance = c
            threshold_index = i
    
    cluster_name = [0]*n
    
    for i in range(0,threshold_index + 1):
        cluster_name[index[i]] = 1
    
    for i in range(0, n):
        clusters[cluster_name[i]].append(i)
        
    # this part changes the cluster_name from a list to numpy array (This step helps to write easy codes)
    cluster_name = np.asarray(cluster_name)
    cluster_name = np.reshape(cluster_name, (1,n))
    
    cp = cluster_name.flatten().tolist()
    
    
    return clusters, cluster_name


# In[10]:


def get_results(l, n, w):
    
    
    fielder = get_fielder(l, n)
    add_col_ncut = []
    add_col_con = []
    
    cl, clp = get_clusters(fielder, w, n) 
    nc_1, c_1 = ncut(clp, w, n)
    
    
    add_col_ncut.append(nc_1)
    add_col_con.append(c_1)
    
    return add_col_ncut, add_col_con


# In[11]:



def unnormalized_results(graph):
    n = np.size(graph, axis = 1)
    
    w = graph

    l = get_unnormalized_laplacian(w)
    return get_results(l, n, w)


# In[15]:


def symmetric_results(graph):

    n = np.size(graph, axis = 1)
    w = graph

    l = get_symmetric_laplacian(w)
    
    
    return get_results(l, n, w)
   


# In[16]:


def random_walk_results(graph):

    n = np.size(graph, axis = 1)
    
    w = graph

    l = get_random_walk_laplacian(w)
    
    
    return get_results(l, n, w)


# In[17]:


def show_results_kannan(graph):
    
    add_col_ncut, add_col_con  = [], []
    
    add_col_ncut, add_col_con = symmetric_results(graph)
    un, uc = unnormalized_results(graph)
    rwn, rwc = random_walk_results(graph)
    
    add_col_ncut += un
    add_col_con += uc
    
    add_col_ncut += rwn
    add_col_con += rwc
    
    return add_col_ncut, add_col_con

