import numpy as np
from tensorflow.keras.datasets import mnist  # just using tensorflow for the mnist dataset
import matplotlib.pyplot as plt

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_net import NeuralNetwork

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshape the images to 2D array (num_samples, num_features)
train_images = train_images.reshape(-1, 28 * 28) / 255.0
test_images = test_images.reshape(-1, 28 * 28) / 255.0

# One-hot encode the labels
train_labels = np.eye(10)[train_labels]
test_labels = np.eye(10)[test_labels]

# Define the number of hidden neurons
hidden_neurons = 70

# Initialize the neural network
nn = NeuralNetwork(train_images, train_labels, hidden_neurons)

# Train the neural network
nn.train(train_images, train_labels, iterations=200, learning_rate=0.000026)

# Evaluate the neural network
predictions = nn.think(test_images)
accuracy = np.mean(np.argmax(predictions, 
axis=1) == np.argmax(test_labels, axis=1))
print(f'Test accuracy: {accuracy * 100:.2f}%')
