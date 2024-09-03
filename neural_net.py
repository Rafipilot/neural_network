import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():
    
    def __init__(self, input, output):

        x, input_length = np.shape(input)
        y, output_length = np.shape(output)

        print("i, o: ", input_length, output_length)
        np.random.seed(1)
        # Initialize weights randomly with mean 0
        self.synaptic_weights = 2 * np.random.random((input_length, output_length)) - 1
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
    




