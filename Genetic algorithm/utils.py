# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 18:53:23 2021

@author: chich
"""
from networkx.algorithms.community.quality import modularity
import random as rdm
# =============================================================================
# Utils script for all functions used is main
# =============================================================================
def fitness_dic(G,pop,numNode):
    # Fitness function with a dictionnary as input
    fitlist = []
    for i in range(len(pop)):
        left = []
        right = []
        for j in range(numNode):
            if pop['C'+str(i)][j] == 0:
                left.append(str(j+1))
            else:
                right.append(str(j+1))            
        modu = modularity(G,[left,right])
        fitlist.append(modu)
    return fitlist


def fitness_list(G,krdm,numNode):
    # Fitness function with a list as input
    fitList = []
    for i in range(len(krdm)):
        left = []
        right = []
        for j in range(numNode):
            if krdm[i][1][j] == 0:
                left.append(str(j+1))
            else:
                right.append(str(j+1))                
        modu = modularity(G,[left,right])
        fitList.append(modu)
    return fitList
    



def selecParents(G,pop,numNode,k):
    # Select best parents with tournament method, here we choose 2 best from K
    # random parents
    Best = []
    krdm = rdm.choices(list(pop.items()),k=k)
    modList = fitness_list(G,krdm,numNode)
    bestKrdm = max(modList)
    index1 = krdm[modList.index(bestKrdm)][0]    
    Best.append(krdm[modList.index(bestKrdm)][1])

    
    krdm = rdm.choices(list(pop.items()),k=k)
    modList = fitness_list(G,krdm,numNode)
    bestKrdm = max(modList)    
    Best.append(krdm[modList.index(bestKrdm)][1])
    index2 = krdm[modList.index(bestKrdm)][0]  

    index = [index1,index2]
    return Best,index

def cross(selection,crossProba):
    # We use uniform crossover with a crossover probability to get the childrens
    p1 = selection[0]
    p2 = selection[1]
    # Point to same 
    c1 = p1.copy()
    c2 = p2.copy()
    for i in range(len(c1)):
        if rdm.random() < crossProba:
            c1[i] = p2[i]
            c2[i] = p1[i]
        # if rdm.random() < crossProba:
        #     c2[i] = p1[i]

    return c1,c2

def mutat(crossover,mutProba):
    # Each gene mutate randomly with mutation proba
    c1 = crossover[0]
    c2 = crossover[1]
    for i in range(len(c1)):
        if rdm.random() < mutProba:
            if c1[i] == 1:
                c1[i] == 0
            elif c1[i] == 0:
                c1[i] == 1
        if rdm.random() < mutProba:
            if c2[i] == 1:
                c2[i] == 0
            elif c2[i] == 0:
                c2[i] == 1
    
    return c1,c2



















    
    