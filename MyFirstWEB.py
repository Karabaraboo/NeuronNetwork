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


def feedforward(self, a):
        '''Вернуть выходные данные сети при входных данных "a"'''
        for b, w in zip(self.biases, self.weights):
            # d = np.dot(w, a)      # Multiplication of matrix (by neurons in layer)
                        # 'a' as matrix (n,1). Else if 'a' is vector (n,) then
                        # 'd' becomes vector. And then sum(np.dot + b) won't be right
            # db = np.dot(w, a)+b   # Plus biase
            # s = sigmoid(db)
            a = sigmoid(np.dot(w, a)+b)
        return a

np.random.seed(10)



net = Network([2,3,1])

print(net.biases)
print(net.weights)

print(feedforward(net,np.array([0,1]).reshape(2,1)))
