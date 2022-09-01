import mnist_loader # For Mnist downloading

import network

# Downloading of Mnist_data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Creating of Network
'''
An input pictures are 28*28 pixels. Therefor an input layer consist of 784 neurons.
30 neurons of hidden layer
10 neurons in output layer. Each neuron corresponds to decimal number from 0 to 9
'''
net = network.Network([784, 30, 10])

# Trainig of Network
'''
30 epochs,
mini_branch size is 10
training velocity is 3.0
'''
net.SGD(training_data, 30, 10, 3.0, test_data = test_data)
