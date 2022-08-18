# My First Network

# Import libraries

import random

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

# Returning of output data if input data is 'a'

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

# Stochastic gradient descent

def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        
        """Обучаем сеть при помощи мини-пакетов и стохастического градиентного спуска. 
        training_data – список кортежей "(x, y)", обозначающих обучающие входные данные и 
        желаемые выходные. 
        Остальные обязательные параметры говорят сами за себя. 
        Если test_data задан, тогда сеть будет оцениваться относительно проверочных данных 
        после каждой эпохи, и будет выводиться текущий прогресс. 
        Это полезно для отслеживания прогресса, однако существенно замедляет работу. """
        
        # training_data     - the list of tuples (x,y) where 
            # 'x' is an input data, 'y' - an expected output data
        # epochs            - a number of epochs for training
        # mini_batch_size   - a size of mini groups of data for training
        # eta               - a velocity of training
        # test_data         - a data for a progress tracking
        
        if test_data: n_test = len(test_data)
        n = len(training_data)
        
        for j in range(epochs):
            random.shuffle(training_data)       # Just shuffling of data 'training_data'
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
                
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
                
            else:
                print("Epoch {0} complete".format(j))


np.random.seed(10)



net = Network([2,3,1])

print(net.biases)
print(net.weights)

print(feedforward(net,np.array([0,1]).reshape(2,1)))
