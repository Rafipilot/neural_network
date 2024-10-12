import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layer_sizes, activation='relu', output_activation='sigmoid'):
        """
        Initialize the neural network.
        
        :param layer_sizes: List of integers representing the number of neurons in each layer.
                            Example: [input_size, hidden1_size, hidden2_size, ..., output_size]
        :param activation: Activation function for hidden layers ('relu' or 'sigmoid').
        :param output_activation: Activation function for the output layer ('sigmoid', etc.).
        """
        # Store the size of each layer
        self.layer_sizes = layer_sizes
        # Total number of layers (including input and output)
        self.num_layers = len(layer_sizes)
        # Activation function for hidden layers
        self.activation = activation
        # Activation function for the output layer
        self.output_activation = output_activation
        
        # Set seed for reproducibility of random numbers
        np.random.seed(1)
        
        # Initialize lists to store weights and biases for each layer
        self.weights = []  # List to hold weight matrices
        self.biases = []   # List to hold bias vectors
        
        # Iterate through each layer to initialize weights and biases
        for i in range(self.num_layers - 1):
            # He initialization for weights: random normal values scaled by sqrt(1 / number of inputs)
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(1. / layer_sizes[i])
            # Biases initialized to zeros
            bias = np.zeros(layer_sizes[i + 1])
            # Append initialized weights and biases to their respective lists
            self.weights.append(weight)
            self.biases.append(bias)
    
    def sigmoid(self, x):
        """
        Compute the sigmoid activation function.
        
        :param x: Input array.
        :return: Sigmoid of input.
        """
        # Clip input to prevent overflow in exponential
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
        Compute the derivative of the sigmoid function.
        
        :param x: Sigmoid output.
        :return: Derivative of sigmoid.
        """
        return x * (1 - x)
    
    def relu(self, x):
        """
        Compute the ReLU activation function.
        
        :param x: Input array.
        :return: ReLU of input.
        """
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """
        Compute the derivative of the ReLU function.
        
        :param x: Input array after ReLU activation.
        :return: Derivative of ReLU.
        """
        return np.where(x > 0, 1, 0)
    
    def activate(self, x, layer): # deciding which activation function to use
        """
        Apply the appropriate activation function based on the layer type.
        
        :param x: Pre-activation input (weighted sum) to the neurons.
        :param layer: Index of the current layer.
        :return: Activated output.
        """
        # Check if current layer is the output layer
        if layer == self.num_layers - 2: 
            if self.output_activation == 'sigmoid':
                return self.sigmoid(x)
            elif self.output_activation == 'relu':
                return self.relu(x)
            else:
                # Raise error if unsupported activation function is specified
                raise NotImplementedError("Output activation not implemented.")
        else:
            # For hidden layers, apply the specified activation function
            if self.activation == 'relu':
                return self.relu(x)
            elif self.activation == 'sigmoid':
                return self.sigmoid(x)
            else:
                # Raise error if unsupported activation function is specified
                raise NotImplementedError("Activation function not implemented.")
    
    def activate_derivative(self, activated, layer):  # deciding which activation derivitive function to use
        """
        Compute the derivative of the activation function based on the layer type.
        
        :param activated: Activated output from the neurons.
        :param layer: Index of the current layer.
        :return: Derivative of the activation function.
        """
        # Check if current layer is the output layer
        if layer == self.num_layers - 2:
            if self.output_activation == 'sigmoid':
                return self.sigmoid_derivative(activated)
            elif self.output_activation == 'relu':
                return self.relu_derivative(activated)
            else:
                # Raise error if derivative for the specified activation is not implemented
                raise NotImplementedError("Output activation derivative not implemented.")
        else:
            # For hidden layers, compute derivative based on the specified activation function
            if self.activation == 'relu':
                return self.relu_derivative(activated)
            elif self.activation == 'sigmoid':
                return self.sigmoid_derivative(activated)
            else:
                # Raise error if derivative for the specified activation is not implemented
                raise NotImplementedError("Activation derivative not implemented.")
    
    def train(self, training_inputs, training_outputs, iterations, learning_rate):
        """
        Train the neural network using backpropagation.
        
        :param training_inputs: List or array of input data for training.
        :param training_outputs: List or array of expected output data.
        :param iterations: Number of training iterations (epochs).
        :param learning_rate: Learning rate for weight and bias updates.
        """
        # Initialize list to store mean absolute error for each iteration
        errors = []
        # Convert training data to NumPy arrays for efficient computation
        training_inputs = np.array(training_inputs)
        training_outputs = np.array(training_outputs)
        
        # Iterate over the specified number of training iterations
        for i in range(iterations):
            # ---------------------
            # 1. Forward Pass
            # ---------------------
            # Initialize list to store activations for each layer; start with input data
            activations = [training_inputs]
            # Initialize list to store pre-activation values (z) for each layer
            pre_activations = []
            
            # Iterate through each layer to compute activations
            for l in range(self.num_layers - 1):
                # Compute pre-activation: z = activation * weights + bias
                z = np.dot(activations[-1], self.weights[l]) + self.biases[l]
                pre_activations.append(z)
                # Apply activation function to z to get activation for next layer
                a = self.activate(z, l)
                activations.append(a)
            
            # ---------------------
            # 2. Compute Error at Output
            # ---------------------
            # Calculate error as difference between expected and actual output
            error = training_outputs - activations[-1]
            # Compute mean absolute error and append to errors list for monitoring
            errors.append(np.mean(np.abs(error)))
            
            # Every 5 iterations, print the current iteration and error
            if i % 5 == 0:
                print(f"Iteration {i}, Error: {errors[-1]}")
            
            # ---------------------
            # 3. Backpropagation
            # ---------------------
            # Initialize list to store delta values for each layer; None as placeholders
            deltas = [None] * (self.num_layers - 1)
            
            # a. Compute delta for the output layer
            # Formula: delta = error * activation_derivative
            deltas[-1] = error * self.activate_derivative(activations[-1], self.num_layers - 2)
            
            # b. Compute deltas for hidden layers (from last hidden to first)
            for l in range(self.num_layers - 3, -1, -1):
                # Propagate delta from the next layer backwards
                # Formula: delta = (delta_next * weights_next.T) * activation_derivative
                deltas[l] = deltas[l + 1].dot(self.weights[l + 1].T) * self.activate_derivative(activations[l + 1], l)
            
            # ---------------------
            # 4. Update Weights and Biases
            # ---------------------
            for l in range(self.num_layers - 1):
                # Input to the current layer is the activation from the previous layer
                layer_input = activations[l]
                # Delta for the current layer
                delta = deltas[l]
                # Update weights: weight += learning_rate * (input.T dot delta)
                self.weights[l] += learning_rate * layer_input.T.dot(delta)
                # Update biases: bias += learning_rate * sum of delta across samples
                self.biases[l] += learning_rate * np.sum(delta, axis=0)
        
        # ---------------------
        # 5. Post-Training Visualization
        # ---------------------
        # Plot the mean absolute error over all training iterations
        plt.plot(errors)
        plt.xlabel('Training Iterations')
        plt.ylabel('Mean Absolute Error')
        plt.title('Training Error Over Time')
        plt.show()
    
    def think(self, inputs):
        """
        Perform a forward pass through the network to generate predictions.
        
        :param inputs: List or array of input data for which to make predictions.
        :return: Network's output after forward pass.
        """
        # Convert inputs to NumPy array of type float for computation
        inputs = np.array(inputs).astype(float)
        
        # Iterate through each layer to compute activations
        for l in range(self.num_layers - 1):
            # Compute pre-activation: z = activation * weights + bias
            z = np.dot(inputs, self.weights[l]) + self.biases[l]
            # Apply activation function to get activation for next layer
            inputs = self.activate(z, l)
        
        # Return the final activation as the output
        return inputs
