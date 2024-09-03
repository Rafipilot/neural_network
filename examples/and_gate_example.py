import numpy as np

import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_net import NeuralNetwork

# Training dataset: 
training_inputs = np.array([[1, 1], 
                        [1, 0], 
                        [0, 1], 
                        [0, 0]]) 


training_outputs = np.array([[1],  
                            [0],  
                            [0],  
                            [0]]) 

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