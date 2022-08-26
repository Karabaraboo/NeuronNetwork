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
                self.update_mini_batch(mini_batch, eta)     # One step of SGD
                
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
                
            else:
                print("Epoch {0} complete".format(j))

    # Update of weights and biases for mini_batch

    def update_mini_batch(self, mini_batch, eta):
        """Обновить веса и смещения сети, применяя градиентный спуск 
        с использованием обратного распространения к одному мини-пакету. 
        mini_batch – это список кортежей (x, y), 
        а eta – скорость обучения."""
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # Creation of array of b.shape dimension
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)  # Sum of derivatives dC/db or dC/dw
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        self.weights = [w-(eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]

    # Back propagation algorithm

    def backprop(self, x, y):
        """Вернуть кортеж ``(nabla_b, nabla_w)``, представляющий градиент 
        для функции стоимости C_x.  
        ``nabla_b`` и ``nabla_w`` - послойные списки массивов numpy, 
        похожие на ``self.biases`` and ``self.weights``."""
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # прямой проход
        activation = x
        activations = [x] # список для послойного хранения активаций
        zs = [] # список для послойного хранения z-векторов
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # обратный проход
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])   # Calculation of derivative of sigmoid
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        """Переменная l в цикле ниже используется не так, как описано во второй главе книги. 
        l = 1 означает последний слой нейронов, 
        l = 2 – предпоследний, и так далее. 
        Мы пользуемся преимуществом того, что в python 
        можно использовать отрицательные индексы в массивах."""
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        """Вернуть количество проверочных входных данных, для которых нейросеть выдаёт правильный результат. Выходные данные сети – это номер нейрона в последнем слое с наивысшим уровнем активации."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Вернуть вектор частных производных (чп C_x / чп a) для выходных активаций."""
        return (output_activations-y)
    

# Definition of function 'sigmoid'

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Derivative of sigmoid 

def sigmoid_prime(z):
    """Производная сигмоиды."""
    return sigmoid(z)*(1-sigmoid(z))
    

# np.random.seed(10)



# net = Network([2,3,1])

# print(net.biases)
# print(net.weights)

# print(net.feedforward(net,np.array([0,1]).reshape(2,1)))
