import numpy as np
import neuron as n

class NeuralNetwork:
    '''
    A neural network with:
      - 2 inputs
      - a hidden layer with 2 neurons (h1, h2)
      - an output layer with 1 neuron (o1)
    Each neuron has the same weights and bias:
      - w = [0, 1]
      - b = 0
    '''
    def __init__(self):
        weights = np.array([0,1])
        bias = 0

        self.h1 = n.Neuron(weights, bias)
        self.h2 = n.Neuron(weights, bias)
        self.o1 = n.Neuron(weights, bias)

    def nw_feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        # The input for out_o1 are the outputs of h1 and h2
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1

def main():
    network = NeuralNetwork()
    x = np.array([2,3])
    print("Neural network output = %s" % network.nw_feedforward(x))


if __name__ == '__main__':
    main()
