import numpy as np
from tensorflow.keras.datasets import mnist  # just using tensorflow for the mnist dataset
import matplotlib.pyplot as plt

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_network.neural_net import NeuralNetwork  # Ensure this refers to the updated NeuralNetwork class

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()



# Reshape the images to 2D array (num_samples, num_features) and normalize
train_images = train_images.reshape(-1, 28 * 28) / 255.0
test_images = test_images.reshape(-1, 28 * 28) / 255.0

# One-hot encode the labels
train_labels = np.eye(10)[train_labels]
test_labels = np.eye(10)[test_labels]

x, input_size = np.shape(train_images) # getting the shape of input
y, output_size = np.shape(train_labels) # getting shape of output

layer_sizes = [input_size, 70, output_size]

# Initialize the neural network
nn = NeuralNetwork(layer_sizes, activation_hidden='relu', activation_output='relu')

# Train the neural network
nn.train(train_images, train_labels, iterations=2000, learning_rate=0.000002)

# Optional: Visualize a few test images and their predictions
num_visualizations = 5
for i in range(num_visualizations):
    plt.imshow(test_images[i].reshape(28, 28), cmap="gray", interpolation="nearest")
    plt.title(f"True Label: {np.argmax(test_labels[i])}")
    plt.show()
    prediction = nn.think(test_images[i].reshape(1, -1))  # Reshape for single sample
    predicted_label = np.argmax(prediction)
    print(f"Predicted Label: {predicted_label}\n")

# Evaluate the neural network on the entire test set
predictions = nn.think(test_images)
accuracy = np.mean(np.argmax(predictions, 
axis=1) == np.argmax(test_labels, axis=1))
print(f'Test accuracy: {accuracy * 100:.2f}%')