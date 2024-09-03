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


training_outputs = np.array([[0],  
                            [1],  
                            [1],  
                            [0]]) 

q = 2
neural_network = NeuralNetwork(training_inputs, training_outputs, q)

neural_network.train(training_inputs, training_outputs, 20000)

while True:
    binary_string = str(input("Input: "))

    binary_input = [int(bit) for bit in binary_string]
    
    print("Considering new situation:", binary_input)
    print("Output:", neural_network.think(np.array([binary_input])))