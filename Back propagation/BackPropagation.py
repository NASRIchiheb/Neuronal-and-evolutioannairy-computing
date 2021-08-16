# -*- coding: utf-8 -*-
"""
@author: chiheb Nasri
"""


from random import random
from math import exp
import csv
import matplotlib.pyplot as plt
import time



#Load the data and create dataset=============================================
def load_txt(filename):
         dataset = []   
         with open(filename, 'r') as f:
            # we skip firsy 4 lines
            for _ in range(4):
                next(f)
            for rows in f:
                # Split rows and transform to floats
                rowy = rows.split()
                map_object = map(float, rowy)
                list_of_integers = list(map_object)
                dataset.append(list_of_integers)
         return dataset

             
#Prepropcessing===============================================================     
 
# Find the min and max values for each column and store it as a tuple
def minmaxFunc(dataset):
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

# Scale dataset data in a [0.1;0.9] range
def dataScale(dataset, minmax):
    scale = [0.1,0.9]
    for row in dataset:
        for i in range(len(row)):              
                row[i] = scale[0] + ((scale[1] - scale[0])/
                                     (minmax[i][1] - minmax[i][0]))*(row[i]-minmax[i][0])
            
def unscale(real,predicted):
    
    for i in range(len(real)):
        real[i] = minmax[4][0]+((minmax[4][1]-minmax[4][0])/
                                (0.9 - 0.1))*(real[i]-0.1)
        predicted[i] = minmax[4][0]+((minmax[4][1]-minmax[4][0])/
                                     (0.9 - 0.1))*(predicted[i]-0.1)    
    return real,predicted
    

# Calculate realative absolut error==========================================
def predError(real, predicted):
    error = 0
    
    for i in range(len(real)):
        numerator = abs(predicted[i]-real[i])
        error += numerator
    final = ((error/sum(real))) * 100  
    
    
    return final





# Feed forward ==============================================================
# get neuron activation for an input
def actFunc(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# We use sigmoid activation function
def sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))

# Forward propagate input pattern to the network
def forward_Propagation(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = actFunc(neuron['weights'], inputs)
            neuron['output'] = sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# get the derivative of an neuron output
def sigmoid_derivative(output):
    return output * (1.0 - output)

# Backpropagate error and store in network
def error_back_propagation(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid_derivative(neuron['output'])

# Update network weights with delta
def update_weights(network, row, learningRate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += learningRate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += learningRate * neuron['delta']


# Train a network for a fixed number of epochs, 
# and calculate prediction error for each epoch
def network_train(network, train, learningRate, numEpoch, numOutputs,test):  
    err_evolution = []
    for epoch in range(numEpoch):
        predictions = list()       
        print("Processing please wait: {}% ".format(int((epoch/numEpoch)*100)))
        print("------------------------------")    
        for row in train:  
            forward_Propagation(network, row)
            expected = [0 for i in range(numOutputs)]
            for i in range(numOutputs):
                expected[i] = row[4]                
            error_back_propagation(network, expected)
            update_weights(network, row, learningRate)
        # Try current network on test set
        for row in test:
            prediction = predict(nn, row)
            predictions.append(prediction)
        real = []
        for i in range(len(test)):
            real.append(test[i][4])
        real, predictions = unscale(real,predictions)        
        error = predError(real, predictions)
        err_evolution.append(error)
    return err_evolution
    
     

# Initialize a network
def init_Network(numInputs, numH_layers, numOutputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(numInputs + 1)]}
                    for i in range(numH_layers)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(numH_layers + 1)]} 
                    for i in range(numOutputs)]
    network.append(output_layer)
    return network

# Make a prediction with a network
def predict(network, row):
    outputs = forward_Propagation(network, row)
    return outputs[0]

# Backpropagation Algorithm
def back_propagation(nn, train, test, learningRate, numEpoch, numH_layers):   
    err_evolution = network_train(nn, train, learningRate, numEpoch, numOutputs,test)    
    predictions = list()
    for row in test:
        prediction = predict(nn, row)
        predictions.append(prediction)
    return predictions,err_evolution
# =============================================================================
def main(train_set, test_set, nn,learningRate,numEpoch,numH_layers):
    
    # Start prediction
    predicted,err_evolution = back_propagation(nn, train_set, test_set,
                                 learningRate, numEpoch, numH_layers)
    # Prediction error on the final version on nn
    real = []
    for i in range(len(test_set)):
        real.append(test_set[i][4])
    real, predicted = unscale(real,predicted)        
    final = predError(real, predicted)
    print("Final relative absolut error: {}% ".format(final))
    
    return real,predicted,err_evolution


# =============================================================================

# # load and prepare data
filename = 'A1-turbine.txt'
dataset = load_txt(filename)

# normalize input variables
minmax = minmaxFunc(dataset)
dataScale(dataset, minmax)

#Split in two sets=========================================================== 
train_set = dataset[:401]
test_set = dataset [401:]

# Init the parametres========================================================
learningRate = 0.2
numEpoch =1000
numH_layers = 10
numInputs = 4
numOutputs= 1

# Count time to execute of the code==========================================
start_time = time.time()

nn = init_Network(numInputs, numH_layers, numOutputs) 

Final_real,Final_predicted,err_evolution = main(train_set, test_set, nn , learningRate,
                      numEpoch, numH_layers)
print("Init parameters: \n Inputs = {}  layers = {}, Output = {} \n learning Rate = {}  Epoch = {}\n".
      format(numInputs,numH_layers,numOutputs, learningRate,numEpoch))

print("--- %s Execution time ---" % (time.time() - start_time))
# We do some plots===========================================================
# Compare Real values and predicted ones---------------------------
line1, = plt.plot(Final_real, label="Real",color = "red")
line2, = plt.plot(Final_predicted, label="Prediction", linestyle='--')

# Create a legend for the first line.
legend = plt.legend(handles=[line1], loc='upper right')

# Add the legend manually to the current Axes.
ax = plt.gca().add_artist(legend)

# Create another legend for the second line.
plt.legend(handles=[line2], loc='lower right')

plt.show()
# -------------------------------------------------------------------------
fig, ax = plt.subplots()
ax.scatter(Final_real,Final_predicted)

ax.set(xlabel='real', ylabel='predicted',
       title="BP : Test data according to predictions")

plt.show()

# Plot evolution of error rate and epoch------------------------------------

fig, ax = plt.subplots()
ax.plot(err_evolution)

ax.set(xlabel='Epochs', ylabel='Relative absolut error',
       title='Evolution of error rate through epochs')
ax.grid(b=True, which='major', color='#666666', linestyle='-')

plt.show()




# Store results in CSV========================================================
header=["Real","Predicted"]
result= [Final_real,Final_predicted]

with open('Results.csv', 'w') as f: 
      
    # using csv.writer method from CSV package 
    write = csv.writer(f) 
      
    write.writerow(header) 
    write.writerows(result)



