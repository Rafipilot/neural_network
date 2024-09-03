import numpy as np

import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_net import NeuralNetwork

# Training dataset: 2x2 images flattened to 4 inputs
training_inputs = np.array([[1, 1, 0, 0],  # Horizontal line
                        [1, 0, 1, 0],  # Vertical line
                        [1, 0, 0, 1],  # Diagonal line
                        [0, 0, 0, 0]]) # No pattern


# Horizontal, Vertical, Diagonal, No pattern
training_outputs = np.array([[1, 0, 0, 0],  # Horizontal line
                            [0, 1, 0, 0],  # Vertical line
                            [0, 0, 1, 0],  # Diagonal line
                            [0, 0, 0, 1]]) # No pattern


neural_network = NeuralNetwork(training_inputs, training_outputs)

print("Random starting weights:", neural_network.synaptic_weights)
print("Random starting bias:", neural_network.bias)

neural_network.train(training_inputs, training_outputs, 20000)

print("Weights after training:", neural_network.synaptic_weights)
print("Bias after training:", neural_network.bias)

while True:
    binary_string = str(input("Input: "))

    binary_input = [int(bit) for bit in binary_string]
    
    print("Considering new situation:", binary_input)
    print("Output:", neural_network.think(np.array([binary_input])))