import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():
    
    def __init__(self, input, output):


        x, input_size = np.shape(input)
        y, output_size = np.shape(output)

        np.random.seed(1)
        # Initialize weights randomly with mean 0
        self.synaptic_weights = 2 * np.random.random((input_size, output_size)) - 1
        self.bias = np.zeros(output_size)  # Adding a bias term for each output neuron

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, iterations):
        errors = []

        for i in range(iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            errors.append(np.mean(np.abs(error)))

            # Calculate the adjustments
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments
            self.bias += np.sum(error * self.sigmoid_derivative(output), axis=0)

        # Plot the error over time after training
        plt.plot(errors)
        plt.xlabel('Training Iterations')
        plt.ylabel('Mean Absolute Error')
        plt.title('Training Error Over Time')
        plt.show()

    def think(self, inputs):
        inputs = inputs.astype(float)
        return self.sigmoid(np.dot(inputs, self.synaptic_weights) + self.bias)


    




