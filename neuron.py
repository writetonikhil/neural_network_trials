import numpy as np

def sigmoid(x):
    print("Running sigmmoid function for x = %s" % x)
    return 1 / (1 + np.exp(-x))   # Sigmoid function S(x) = 1/(1 + e^(-x)) = e^x / (e^x + 1)

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # Multiply weights, add bias and then use the activation function sigmoid
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

def main():
    weights = np.array([0,1])  #w1=0 and w2=1
    bias = 4 # consider it for example
    n = Neuron(weights, bias)

    x = np.array([2,3]) # x1=2, x2=3
    print("Neuron value for input %s = %s" % (x, n.feedforward(x)))

if __name__ == '__main__':
	main()
