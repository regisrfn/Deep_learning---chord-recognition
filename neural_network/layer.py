import numpy as np

class Layer:
    def __init__(self,input_size,outputsize):
        self.w = np.random.randn(input_size, outputsize)
        self.b = np.zeros((1, self.w.shape[1]))
        self.a = np.array(0)