import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():
    
    def __init__(self, input, output, q):
        x, input_size = np.shape(input)
        y, output_size = np.shape(output)
        
        np.random.seed(1)
        # Initialize weights randomly with mean 0
        self.weights_input_hidden = np.random.randn(input_size, q) * np.sqrt(1. / input_size)
        self.weights_hidden_output = np.random.randn(q, output_size) * np.sqrt(1. / q)
        
        self.bias_hidden = np.zeros(q)  # Bias for hidden layer
        self.bias_output = np.zeros(output_size)  # Bias for output layer

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def train(self, training_inputs, training_outputs, iterations, learning_rate):
        errors = []

        for i in range(iterations):
            # Forward pass using ReLU for the hidden layer
            hidden_output = self.relu(np.dot(training_inputs, self.weights_input_hidden) + self.bias_hidden)
            final_output = self.sigmoid(np.dot(hidden_output, self.weights_hidden_output) + self.bias_output)
            
            # Calculate error
            error = training_outputs - final_output
            errors.append(np.mean(np.abs(error)))

            # Backward pass
            final_output_delta = error * self.sigmoid_derivative(final_output)
            hidden_output_error = final_output_delta.dot(self.weights_hidden_output.T)
            hidden_output_delta = hidden_output_error * self.relu_derivative(hidden_output)
            
            # Update weights and biases with learning rate
            self.weights_hidden_output += learning_rate * hidden_output.T.dot(final_output_delta)
            self.weights_input_hidden += learning_rate * training_inputs.T.dot(hidden_output_delta)
            
            self.bias_output += learning_rate * np.sum(final_output_delta, axis=0)
            self.bias_hidden += learning_rate * np.sum(hidden_output_delta, axis=0)

        # Plot the error over time after training
        plt.plot(errors)
        plt.xlabel('Training Iterations')
        plt.ylabel('Mean Absolute Error')
        plt.title('Training Error Over Time')
        plt.show()

    def think(self, inputs):
        inputs = inputs.astype(float)
        hidden_output = self.relu(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        final_output = self.sigmoid(np.dot(hidden_output, self.weights_hidden_output) + self.bias_output)
        return final_output
