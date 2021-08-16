     # -*- coding: utf-8 -*-
"""
@author: chiheb
"""

import networkx as nx
import random as rdm
import utils
import matplotlib as plt
# =============================================================================
# Select parameters
# =============================================================================
dirPath = "A4-networks/"
filePath = "wheel.net"
popsize = 100
# Increase num generation
generation = 250
# Mutation probability
mutProba = 0.05 
# Crossover probability
crossProba = 0.5
# K parents selection tournament method
k = 5

# =============================================================================
# Load the network
# =============================================================================
# Load the graph and store it as nx object
G = nx.Graph(nx.read_pajek(dirPath+filePath))
# Nx ML READER
nx.draw(G,with_labels=True)
plt.pyplot.show()
# =============================================================================
# Initialize population by giving random clusters
# =============================================================================
nodesList = list(G.nodes)
numNode = len(G.nodes())
pop = {}
    
# Create population of popSize chromosomes.
# Seperate the nodes in two clusters randomly and store in C
# Where 0 = left cluster and 1 = right cluster 
for i in range(popsize):
    pop['C'+str(i)] = []
    for j in nodesList:        
        if rdm.random() > 0.5:
            pop['C'+str(i)].append(0)            
        else:  
            pop['C'+str(i)].append(1)

# =============================================================================
# Calculute modularity for all individuals
# =============================================================================
fitList = []
fitList = utils.fitness_dic(G,pop,numNode)
print("best start modularity : ",max(fitList))
# =============================================================================
# Start the algorithm
# =============================================================================

bestResults = []

for i in range(generation):
    print("Generation : ",i)
    # We init children dictionnary
    newPop = {}
    for j in range(int(popsize)):
        selection = utils.selecParents(G,pop,numNode,k)
        crossover = utils.cross(selection[0],crossProba)
        mutation = utils.mutat(crossover,mutProba)
        newPop['C'+str(j)] = mutation[0]
        newPop['C'+str(j)] = mutation[1]
    #Elitism to add
    # We update population
    pop = newPop.copy()
    # # We compute modularity for all of the children
    # fitList = utils.fitness_dic(G,pop,numNode)
    # Save best results for plots
    bestResult = utils.fitness_dic(G,pop,numNode)
    bestResults.append(max(bestResult))
# Get best partition
popModularity = utils.fitness_dic(G,pop,numNode)
bestModulrarity = max(popModularity)
bestModIndex = popModularity.index(bestModulrarity)
bestPartition = pop['C'+str(bestModIndex)]

# =============================================================================
# Some plots
# =============================================================================
# Evolution of modularity trhough iteration
print("best modularity at the end : ",max(bestResults))    
plt.pyplot.plot(bestResults)
plt.pyplot.xlabel("Iteration")
plt.pyplot.ylabel("Fitness")
plt.pyplot.show()        
#Plot the graph with cluuster
color_map = []
for node in G:
    if bestPartition[int(node)-1] == 1:
        color_map.append('blue')
    else: 
        color_map.append('green')      
nx.draw(G, node_color=color_map, with_labels=True)
plt.pyplot.show()
# =============================================================================
# Save the result in Pajek .clu format
# =============================================================================
# parse input file path to get the name
fileName = filePath.split(".")

with open('output/'+fileName[0]+'.clu', 'w') as f:
    f.write("*Vertices %s\n" %numNode)
    for item in bestPartition:
        f.write("%s\n" % item)

# # =============================================================================
# # Modularity function (beta test)
# # =============================================================================

# def modularityy(G,A,degree):
#     counter = 0       
#     Q = 0
#     l2= (2*G.number_of_edges())
#     for i in range(len(nodesList)):
#         for j in range(len(nodesList)):
#             if i == j:
#                 counter += 1
#             else:                
#                 if str(i+1) in left and str(j+1) in left:
#                     # delta = 1
#                     # Vector of degreees
#                     q =(A[int(i),int(j)]-((degrees[int(i)]*degrees[int(j)])/l2))
#                     counter += 1 
#                     Q += q  
#                 elif str(j+1) in right and str(j+1) in right:
#                     # delta = 1
#                     q =(A[int(i),int(j)]-((degrees[int(i)]*degrees[int(j)])/l2))
#                     counter += 1
#                     # Missing factor
#                     Q += q  
#                 # else:       
#                 #     delta = 0
#                 #     q =(A[int(i),int(j)]-((degrees[int(i)]*degrees[int(j)])/l2))*delta 
#                 #     Q += (1/l2)*q  
#                 #     counter += 1
#     Q = (1/l2) * Q
#     return Q
























