#!/usr/bin/env python
# coding: utf-8

# This code has all algorithm which we proposed and we check it using two types of laplacian that is symmetic and unnormalized laplacian. 
# And we have set the precision for the edge weigths here so we get desired results for the Petersen graph.
# Here we have used edge weigths as (f_i - f_j)^2. 
# 
# And we are doing it for just for graphs with edge weigth 1 or 0.
# 
# # here i have added the code for random walk laplacian 
# 

# In[1]:


import numpy as np
import scipy as sp
from numpy import linalg as LA
import networkx.linalg.algebraicconnectivity as alg
import collections 
import math
import pandas as pd


# In[2]:


# In[3]:


def get_unnormalized_laplacian(w):
    
    D = w.sum(axis = 1)
    
    D = np.diag(D)  # The degree matrix D
    
    L = D - w
    
    return L


# In[4]:


def get_symmetric_laplacian(w):
    
    D = w.sum(axis = 1)
    zero_entries = np.where(D == 0)[0]
    D[zero_entries] = 1
    D_sqrt = np.sqrt(D)
    
    D_1 = np.reciprocal(D_sqrt)
    D_1 = np.diag(D_1)  
    D = np.diag(D)
    L = np.dot(D_1, np.dot(D - w, D_1))
    
    return L


# In[5]:


def get_random_walk_laplacian(w):
    
    D = w.sum(axis = 1)
    zero_entries = np.where(D == 0)[0]
    D[zero_entries] = 1
    D_1 = np.reciprocal(D)
    D_1 = np.diag(D_1)  
    D = np.diag(D)
    L = np.dot(D - w, D_1)
    
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
        
    fielder = eigvecs.T[1:cardinality, :] #The 2nd smallest eigen vector of the laplacian L = D - W
    
    return fielder


# In[7]:


# Checks weather the graph has to connected components or not i.e is the garph paritioned into two clusters 

def check_partion(w, n): # Similarity matrix W and the number of vetrices n
    
    # We will do bfs and check if the graph is partitioned into two parts or not 
    explored = [0]*n 
    
    queue = collections.deque([0])
    
    queue_size = 1
    
    while(queue_size != 0):
        
        node = queue.popleft()
        queue_size -= 1
        
        if explored[node] == 0:
            
            explored[node] = 1
 
            for i in range(0, n):
                if(w[node][i] != 0):
                    neighbour = i
                    
                    if(explored[neighbour] == 0):
                        queue.append(neighbour)
                        queue_size += 1
    
    # Just need to check if this graph has been partitioned into two parts or not 
    is_partitioned = False
    
    for node in explored:
        if node == 0:
            is_partitioned = True
    
    return is_partitioned  

    


# In[8]:


# If the graph has been paritioned into two parts we need to know what these partitions are this function gives us that
# not only does it outputs the clusters but also a vector whose ith index is 1 if its in the first cluster otherwise 0.

def get_partition(w, n): # the similrity matrix W and the number of vertices n
    
    # We will do bfs from a vertex and find out its conected component the other vertices which are not in the
    # connected component are in the other cluster 
    clusters = [[], []]
    explored = [0]*n
    
    
    queue = collections.deque([0])
    
    queue_size = 1
    
    
    while(queue_size != 0):
        
        node = queue.popleft()
        queue_size -= 1
        
        if explored[node] == 0:
            
            explored[node] = 1
 
            for i in range(0, n):
                if(w[node][i] != 0):
                    neighbour = i
                    
                    if(explored[neighbour] == 0):
                        queue.append(neighbour)
                        queue_size += 1
    
    
    #If it a node is connected to vertex zero then its in cluster 0 otherwise its in cluster 1. 
    for node in range(0, n):
        if explored[node] == 1:
            clusters[0].append(node)
        else:
            clusters[1].append(node)
    
    return clusters, explored


# In[9]:


def get_edge_weights(f, w, n):
    
    new_weight = []
    D = w.sum(axis = 1)
    D_sqrt = np.sqrt(D)
    
    for i in range(0, n):
        for j in range(0, n):
            if(w[i][j] != 0):
                new_weight.append([np.dot((f[:,i] - f[:,j]).T,f[:,i] - f[:,j]), i, j])
    
    new_weight = [ [round(i, 3) for i in elem] for elem in new_weight]
    new_weight = sorted(new_weight, reverse= True)
    

    return new_weight


# In[10]:


# Here we use our current algorithm to partition the graph
# This returns the clusters, the cluster_name vector and the edeges cut
def get_clusters(f, w, n): # the similarity matrix and the number of vertices
    
    new_weight = get_edge_weights(f, w, n)
    
    new_weight.sort(reverse= True)

    new_weight = collections.deque(new_weight)

    new_w = w.copy() # Here we copy the similarity matrix so that when we make changes to new_w nothing happens to the origianl similarty matrix

    edges_cut = []
    while(check_partion(new_w, n) == False):  # keep on removing edges until we have a parition
        edge_remove = new_weight.popleft()

        u = edge_remove[1]
        v = edge_remove[2]

        if(new_w[u][v] != 0):
            edges_cut.append([u,v])

        new_w[u][v] = 0
        new_w[v][u] = 0
        
    
    clusters, cluster_name = get_partition(new_w, n) 
    
    # this part changes the cluster_name from a list to numpy array (This step helps to write easy codes)
    cluster_name = np.asarray(cluster_name)
    cluster_name = np.reshape(cluster_name, (1,n))
    
    
    return clusters, cluster_name, edges_cut 


# In[11]:


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

    


# In[12]:


def get_cut(clusters, w, n):
    cut = []
    for x in clusters[0]:
        for y in clusters[1]:
            if(w[x][y] != 0):
                cut.append([x,y])
    return cut


# In[13]:
def hypothesis(edges_cut, cluster_predict, w, n):
    
    m = len(edges_cut)
    
    w_copy = w.copy()
    for i in range(0, n):
        for j in range(0, n):
            if w[i][j] > 0:
                w[i][j] = 1
                
    deg = w.sum(axis = 1)
    
    
    cluster_order = [[],[]]
    
    for i in range(0, n):
        for j in range(0, n):
            if(cluster_predict[0][i] == cluster_predict[0][j]):
                deg[i] -= w[i][j]
                
    for i in range(0, n):
        if deg[i] == 0:
            cluster_order[cluster_predict[0][i]].append(i)
            
    for i in range(0, m):
        #print(cluster_order)
        [x, y] = edges_cut[i]
        
        if cluster_predict[0][x] != cluster_predict[0][y]:
            deg[x] -= w[x][y]
            deg[y] -= w[y][x]
        
            if(deg[x] == 0):
                cluster_order[cluster_predict[0][x]].append(x)
            if(deg[y] == 0):
                cluster_order[cluster_predict[0][y]].append(y)
    
    order = []
    
    for x in cluster_order[0]:
        order.append(x)
    
    n_1 = len(cluster_order[1])
    
    for x in range(1, n_1 + 1):
        order.append(cluster_order[1][n_1 - x])
    
    threshold_index = 0
    best_conductance = 2
    cluster_name = [0]*n
    clusters = [[], []]
    
    w = w_copy
    for i in range(0, n - 1):
        
        cluster_name[order[i]] = 1
        nc, c = ncut(cluster_name, w, n)
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
    
    return clusters, cluster_name
    
def get_clusters_best_index(f, w, n): # the similarity matrix and the number of vertices
    
    threshold_index = 0
    index = np.argsort(f)
    index = index[0]
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

def get_results(l, n, w):
    
    fielder = get_fielder(l, n)
    add_col_ncut = []
    add_col_con = []
    
    cl, clp = get_clusters_best_index(fielder, w, n) 
    nc_1, c_1 = ncut(clp, w, n)
    
    add_col_ncut.append(nc_1)
    add_col_con.append(c_1)

    clusters, cluster_predict , edges_cut = get_clusters(np.reshape(fielder[0,:],(1,n)), w, n) 
    
    nc, c = ncut(cluster_predict, w, n)
    add_col_ncut.append(nc)
    add_col_con.append(c)

    cl, clp = hypothesis(edges_cut, cluster_predict, w, n)
    nc_1, c_1 = ncut(clp, w, n)
    
    add_col_ncut.append(nc_1)
    add_col_con.append(c_1)
    
    
    return add_col_ncut, add_col_con

# In[14]:


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


def show_results_hypothesis(graph):
    
    add_col_ncut, add_col_con  = [], []
    
    add_col_ncut, add_col_con = symmetric_results(graph)
    un, uc = unnormalized_results(graph)
    rwn, rwc = random_walk_results(graph)
    
    add_col_ncut += un
    add_col_con += uc
    
    add_col_ncut += rwn
    add_col_con += rwc
    
    return add_col_ncut, add_col_con

