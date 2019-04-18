import numpy as np
from neural_network.layer import Layer

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class NeuralNetwork:
    def __init__(self, x, y):
        self.y = y
        self.x = x
        self.layers = []

    def feedforward(self):
        x = self.x
        for index , layer in enumerate(self.layers):
            y = sigmoid(np.dot(x, layer.w) + layer.b)
            x = y
            self.layers[index].a = y

    def backprop(self, learning_rate=0.1):
        N = len(self.x)
        num_layers = len(self.layers)

        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1

        a_delta = (self.layers[-1].a - self.y)  # w2

        for i in reversed(range(num_layers)):

            if (i-1) < 0:
                a_layer_before = self.x.T
            else:
                a_layer_before = self.layers[i-1].a.T

            if(i != num_layers-1):
                z_delta = np.dot(a_delta, self.layers[i+1].w.T)
                a_delta = z_delta * sigmoid_derivative(self.layers[i].a)  # w

            self.layers[i].w -= learning_rate * np.dot(a_layer_before, a_delta)/N
            self.layers[i].b -= np.sum(a_delta, axis=0)/N


    def predict(self, x_test):
        x = x_test
        for layer in (self.layers):
            y = sigmoid(np.dot(x, layer.w) + layer.b)
            x = y
        return y

    def add_layer(self,size,input_size=0):
        if (input_size != 0):
            last_layer_size = input_size
            new_layer = Layer(input_size=last_layer_size, outputsize=size)
            self.layers.append(new_layer)
        else:
            last_layer_size = self.layers[-1].w.shape[1]
            new_layer = Layer(input_size=last_layer_size, outputsize=size)
            self.layers.append(new_layer)
