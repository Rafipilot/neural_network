import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():
    
    def __init__(self):
        np.random.seed(1)
        # Initialize weights randomly with mean 0
        self.synaptic_weights = 2 * np.random.random((4, 4)) - 1
        self.bias = 2 * np.random.random(1) - 1  # Adding a bias term

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_der(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_i):

        errors = []

        for i in range(training_i):
            output = self.think(training_inputs)
            error = training_outputs - output

            errors.append(np.mean(np.abs(error)))
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_der(output))
            
            # Adjust weights and bias
            self.synaptic_weights += adjustments
            self.bias += np.sum(error * self.sigmoid_der(output))

                # Plot the error over time after training
        plt.plot(errors)
        plt.xlabel('Training Iterations')
        plt.ylabel('Mean Absolute Error')
        plt.title('Training Error Over Time')
        #plt.show()

    def think(self, inputs):
        inputs = inputs.astype(float)
        return self.sigmoid(np.dot(inputs, self.synaptic_weights) + self.bias)
    

if __name__ == "__main__":
    neural_network = NeuralNetwork()
    
    print("Random starting weights:", neural_network.synaptic_weights)
    print("Random starting bias:", neural_network.bias)

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

    neural_network.train(training_inputs, training_outputs, 20000)

    print("Weights after training:", neural_network.synaptic_weights)
    print("Bias after training:", neural_network.bias)

    while True:
        binary_string = str(input("Input: "))

        binary_input = [int(bit) for bit in binary_string]
        
        print("Considering new situation:", binary_input)
        print("Output:", neural_network.think(np.array([binary_input])))
