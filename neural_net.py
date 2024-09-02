import numpy as np

class NeuralNetwork():
    
    def __init__(self):
        # Set the random seed to ensure the same initial weights are generated each time.
        np.random.seed(1)

        
        # Initialize synaptic weights randomly with a mean of 0
        # We have 3 inputs and 1 output, hence the shape (3, 1)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        # Sigmoid activation function: maps any value to a value between 0 and 1
        return 1 / (1 + np.exp(-x))

    def sigmoid_der(self, x):
        # Derivative of the sigmoid function, used to calculate gradients during backpropagation
        # This helps in adjusting the weights effectively
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_i):
        # Training the neural network by adjusting the weights based on the output error
        for i in range(training_i):
            # Pass the training set through the neural network (forward propagation)
            output = self.think(training_inputs)
            
            # Calculate the error (the difference between the desired output and the predicted output)
            error = training_outputs - output
            
            # Backpropagation: Adjusting the weights
            # The adjustment is proportional to the error and the slope of the sigmoid curve (using its derivative)
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_der(output))
            
            # Update the weights with the adjustments
            self.synaptic_weights += adjustments
        
    def think(self, inputs):
        # Pass inputs through the neural network to get the output (forward propagation)
        inputs = inputs.astype(float)
        # The output is the result of applying the sigmoid function to the dot product of inputs and weights
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output
    

if __name__ == "__main__":
    # Initialize a single neuron neural network
    neural_network = NeuralNetwork()
    
    # Display the starting synaptic weights before training
    print("Random starting weights:", neural_network.synaptic_weights)

    # Define training inputs (4 examples with 3 inputs each)
    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]])

    # Define training outputs (desired outputs for each example)
    # .T makes it a column vector, matching the expected shape
    training_outputs = np.array([[0, 1, 1, 0]]).T

    # Train the neural network with the training set, adjusting weights 60,000 times
    neural_network.train(training_inputs, training_outputs, 60000)

    # Display the synaptic weights after training
    print("Weights after training:", neural_network.synaptic_weights)

    # Interactive loop to test the neural network with new inputs
    while True:
        # Get inputs from the user
        A = float(input("Input 1: "))
        B = float(input("Input 2: "))
        C = float(input("Input 3: "))

        # Display the inputs
        print("Considering new situation:", A, B, C)
        
        # Pass the new inputs through the network and display the output
        print("Output:", neural_network.think(np.array([A, B, C])))


    


