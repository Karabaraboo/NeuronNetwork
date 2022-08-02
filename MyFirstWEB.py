# My First Network

# Import libraries

from random import seed
import numpy as np

# Creating a class Network

class Network() : 

    # Method - Constructor
    def __init__(self, sizes) :         # Self - object - Our created Net. Sizes - vector which is given
        self.num_layers = len(sizes)    # Numbers of web layers
        self.sizes = sizes              # Numbers of neurons in corresponding layer
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]     # Offsets for each neuron in each layer
                        # Except first layer.    Standart normal distribution
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]     # Weights for neurons
                        # In rows - neurons of current layer
                        # In columns - neurons of privious layer
    

# Definition of function 'sigmoid'

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

np.random.seed(10)



net = Network([2,3,1])

print(net.biases)
print(net.weights)
