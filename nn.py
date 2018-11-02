import math
import random
import numpy
import matplotlib

class NeuralNetwork:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [numpy.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [numpy.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def calculate(self, input):
        for b, w in zip(self.biases, self.weights):
            input = sigmoid(numpy.dot(w, input) + b)
        return input

    def train(self, training_data, epochs, mini_batch_size, learning_rate, test_data = None):
        n = len(training_data)
        for e in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[index : index + mini_batch_size] for index in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if None != test_data:
                n_test = len(test_data)
                num_correct = self.evaluate(test_data)
                print("Epoch " + e + ": " + num_correct + " out of " + n_test + ".")
            else:
                print("Epoch " + e + " completed.")

    #Learn this part
    def update_mini_batch(self, mini_batch, learning_rate):
        grad_b = [numpy.zeroes(b) for b in self.biases]
        grad_w = [numpy.zeroes(w) for w in self.weights]
        for x, y in mini_batch:
            dgrad_b, dgrad_w = self.backprop(x, y)
            grad_b = [gb + dgb for gb, dgb in zip(grad_b, dgrad_b)]
            grad_w = [gw + dgw for gw, dgw in zip(grad_w, dgrad_w)]
        self.weights = [w - (learning_rate / len(mini_batch)) * gw for w, gw in zip(self.weights, grad_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * gb for b, gb in zip(self.biases, grad_b)]


    def backprop(self, x, y):
        grad_w = [numpy.zeroes(w) for w in self.weights]
        grad_b = [numpy.zeroes(b) for b in self.biases]

        #Feed-forward while keeping track of activations and weighted sums
        a = x
        #Lists store the relevant values for each layer
        a_list = [x]
        z_list = []
        for b, w in zip(self.biases, self.weights):
            z = numpy.dot(w, a) + b
            z_list.append(z)
            a = sigmoid(z)
            a_list.append(a)
        #Backprop to calc the change in weights and biases
        d = (a_list[-1] - y) * dsigmoid(z_list[-1])
        grad_b = d
        grad_w = numpy.dot(d, a_list[-2].transpose())

        #Using layer to go back in the network
        for layer in range(2, self.num_layers):
            z = z_list[-layer]
            ds = dsigmoid(z)
            d = numpy.dot(self.weights[-layer + 1].transpose(), d) * ds
            grad_b[-layer] = d
            grad_w[-layer] = numpy.dot(d, a_list[-layer - 1].transpose())
        return (grad_b, grad_w)
    
    def test(self, test_data):
        test_results = [numpy.argmax(self.calculate(x)), y for x, y in test_data]
        return sum((1 if (x == y) else 0) for (x, y) in test_results)

#Sigmoid and its derivative
def sigmoid(val):
    return 1.0 / (1.0 + numpy.exp(-val))

def dsigmoid(val):
    s = sigmoid(val)
    return s * (1 - s)