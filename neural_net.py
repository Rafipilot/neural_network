import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layer_sizes, activation_hidden='relu', activation_output='sigmoid'):
        """
        Initializes the neural network with a given architecture.

        Parameters:
        - layer_sizes: List of integers specifying the number of neurons in each layer,
                       including input and output layers.
                       Example: [2, 3, 2, 1] represents a network with:
                           - 2 input neurons
                           - 3 neurons in the first hidden layer
                           - 2 neurons in the second hidden layer
                           - 1 output neuron
        - activation_hidden: Activation function for hidden layers ('relu' or 'sigmoid').
        - activation_output: Activation function for the output layer ('sigmoid' or 'relu').
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.activation_hidden = activation_hidden
        self.activation_output = activation_output
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        np.random.seed(1)  # For reproducibility

        for i in range(1, self.num_layers):
            input_size = layer_sizes[i-1]
            output_size = layer_sizes[i]
            # He initialization for ReLU, Xavier for Sigmoid
            if (i != self.num_layers -1 and activation_hidden == 'relu') or (i == self.num_layers -1 and activation_output == 'relu'):
                weight = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
            else:
                weight = np.random.randn(input_size, output_size) * np.sqrt(1. / input_size)
            self.weights.append(weight)
            self.biases.append(np.zeros(output_size))

    def sigmoid(self, x):
        """Sigmoid activation function."""
        x = np.clip(x, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of the sigmoid function."""
        return x * (1 - x)

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of the ReLU function."""
        return np.where(x > 0, 1, 0)
    
    def softmax():
        pass

    def forward_pass(self, inputs):
        """
        Performs a forward pass through the network.

        Parameters:
        - inputs: Input data as a NumPy array.

        Returns:
        - activations: List of activations for each layer.
        """
        activations = [inputs]
        for i in range(self.num_layers - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            if i < self.num_layers - 2:
                # Hidden layers
                if self.activation_hidden == 'relu':
                    a = self.relu(z)
                elif self.activation_hidden == 'sigmoid':
                    a = self.sigmoid(z)
                else:
                    raise ValueError("Unsupported activation function for hidden layers.")
            else:
                # Output layer
                if self.activation_output == 'sigmoid':
                    a = self.sigmoid(z)
                elif self.activation_output == 'relu':
                    a = self.relu(z)
                else:
                    raise ValueError("Unsupported activation function for output layer.")
            activations.append(a)
        return activations

    def backpropagate(self, activations, training_outputs, learning_rate):
        """
        Performs backpropagation and updates the weights and biases.

        Parameters:
        - activations: List of activations from the forward pass.
        - training_outputs: True output values.
        - learning_rate: Learning rate for weight updates.
        """
        # Initialize list to store deltas
        deltas = [None] * (self.num_layers -1)

        # Compute delta for output layer
        error = training_outputs - activations[-1]
        if self.activation_output == 'sigmoid':
            delta = error * self.sigmoid_derivative(activations[-1])
        elif self.activation_output == 'relu':
            delta = error * self.relu_derivative(activations[-1])
        deltas[-1] = delta

        # Compute deltas for hidden layers (in reverse order)
        for i in reversed(range(self.num_layers - 2)):
            if self.activation_hidden == 'relu':
                delta = deltas[i+1].dot(self.weights[i+1].T) * self.relu_derivative(activations[i+1])
            elif self.activation_hidden == 'sigmoid':
                delta = deltas[i+1].dot(self.weights[i+1].T) * self.sigmoid_derivative(activations[i+1])
            deltas[i] = delta

        # Update weights and biases
        for i in range(self.num_layers -1):
            layer_input = activations[i]
            delta = deltas[i]
            # Update weights
            self.weights[i] += learning_rate * np.dot(layer_input.T, delta)
            # Update biases
            self.biases[i] += learning_rate * np.sum(delta, axis=0)

    def train(self, training_inputs, training_outputs, iterations, learning_rate):
        """
        Trains the neural network using the provided training data.

        Parameters:
        - training_inputs: Input data as a NumPy array.
        - training_outputs: True output values as a NumPy array.
        - iterations: Number of training iterations.
        - learning_rate: Learning rate for weight updates.
        """
        errors = []
        for i in range(iterations):
            # Forward pass
            activations = self.forward_pass(training_inputs)
            output = activations[-1]

            # Compute error
            error = training_outputs - output
            mean_error = np.mean(np.abs(error))
            errors.append(mean_error)

            # Print error every 1000 iterations
            if (i % 1000 == 0) or (i == iterations -1):
                print(f"Iteration {i+1}/{iterations}, Error: {mean_error}")

            # Backpropagation
            self.backpropagate(activations, training_outputs, learning_rate)

        # Plot the error over time after training
        plt.plot(errors)
        plt.xlabel('Training Iterations')
        plt.ylabel('Mean Absolute Error')
        plt.title('Training Error Over Time')
        plt.grid(True)
        plt.show()

    def think(self, inputs):
        """
        Performs a forward pass and returns the network's output.

        Parameters:
        - inputs: Input data as a NumPy array.

        Returns:
        - final_output: Network's output.
        """
        activations = self.forward_pass(inputs)
        return activations[-1]
    
    def think_batch(self, inputs, batch_size=1000):
        """
        Performs a forward pass on the input data in batches.

        Parameters:
        - inputs: Input data as a NumPy array.
        - batch_size: Number of samples to process at a time.

        Returns:
        - final_output: Network's output for all inputs.
        """
        num_samples = inputs.shape[0]
        final_output = []
        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            batch_inputs = inputs[start:end]
            batch_output = self.forward_pass(batch_inputs)[-1]
            final_output.append(batch_output)
        return np.vstack(final_output)

