import neural_network.neural_net as neural_net


# Initialize the network
nn = neural_net.NeuralNetwork(layer_sizes=[3, 5, 1], activation='relu', output_activation='sigmoid')

# Training data
training_inputs = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]
training_outputs = [[0], [1], [1], [0]]

# Train the network
nn.train(training_inputs, training_outputs, iterations=10000, learning_rate=0.01)

# Make a prediction
new_input = [1, 0, 1]
output = nn.think(new_input)
print(f"Prediction for {new_input}: {output}")