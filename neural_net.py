import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layer_sizes, activation='relu', output_activation='sigmoid'):
        """
        Initialize the neural network.
        
        :param layer_sizes: List of integers representing the number of neurons in each layer.
                            Example: [input_size, hidden1_size, hidden2_size, ..., output_size]
        :param activation: Activation function for hidden layers ('relu' or 'sigmoid').
        :param output_activation: Activation function for the output layer ('sigmoid', 'softmax', etc.).
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.activation = activation
        self.output_activation = output_activation
        
        np.random.seed(1)  # For reproducibility
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        for i in range(self.num_layers - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(1. / layer_sizes[i])
            bias = np.zeros(layer_sizes[i + 1])
            self.weights.append(weight)
            self.biases.append(bias)
    
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def activate(self, x, layer):
        if layer == self.num_layers - 2:  # Output layer
            if self.output_activation == 'sigmoid':
                return self.sigmoid(x)
            elif self.output_activation == 'relu':
                return self.relu(x)
            else:
                raise NotImplementedError("Output activation not implemented.")
        else:  # Hidden layers
            if self.activation == 'relu':
                return self.relu(x)
            elif self.activation == 'sigmoid':
                return self.sigmoid(x)
            else:
                raise NotImplementedError("Activation function not implemented.")
    
    def activate_derivative(self, activated, layer):
        if layer == self.num_layers - 2:  # Output layer
            if self.output_activation == 'sigmoid':
                return self.sigmoid_derivative(activated)
            elif self.output_activation == 'relu':
                return self.relu_derivative(activated)
            else:
                raise NotImplementedError("Output activation derivative not implemented.")
        else:  # Hidden layers
            if self.activation == 'relu':
                return self.relu_derivative(activated)
            elif self.activation == 'sigmoid':
                return self.sigmoid_derivative(activated)
            else:
                raise NotImplementedError("Activation derivative not implemented.")
    
    def train(self, training_inputs, training_outputs, iterations, learning_rate):
        errors = []
        training_inputs = np.array(training_inputs)
        training_outputs = np.array(training_outputs)
        
        for i in range(iterations):
            # Forward pass
            activations = [training_inputs]
            pre_activations = []
            for l in range(self.num_layers - 1):
                z = np.dot(activations[-1], self.weights[l]) + self.biases[l]
                pre_activations.append(z)
                a = self.activate(z, l)
                activations.append(a)
            
            # Compute error at output
            error = training_outputs - activations[-1]
            errors.append(np.mean(np.abs(error)))
            
            if i % 5 == 0:
                print(f"Iteration {i}, Error: {errors[-1]}")
            
            # Backpropagation
            deltas = [None] * (self.num_layers - 1)
            # Output layer delta
            deltas[-1] = error * self.activate_derivative(activations[-1], self.num_layers - 2)
            
            # Hidden layers delta
            for l in range(self.num_layers - 3, -1, -1):
                deltas[l] = deltas[l + 1].dot(self.weights[l + 1].T) * self.activate_derivative(activations[l + 1], l)
            
            # Update weights and biases
            for l in range(self.num_layers - 1):
                layer_input = activations[l]
                delta = deltas[l]
                self.weights[l] += learning_rate * layer_input.T.dot(delta)
                self.biases[l] += learning_rate * np.sum(delta, axis=0)
        
        # Plot the error over time after training
        plt.plot(errors)
        plt.xlabel('Training Iterations')
        plt.ylabel('Mean Absolute Error')
        plt.title('Training Error Over Time')
        plt.show()
    
    def think(self, inputs):
        inputs = np.array(inputs).astype(float)
        for l in range(self.num_layers - 1):
            inputs = self.activate(np.dot(inputs, self.weights[l]) + self.biases[l], l)
        return inputs
