# Neural Network from Scratch in Python

This project is an implementation of a simple feedforward neural network using Python and NumPy. The network supports multiple hidden layers, and the activation functions available are ReLU and Sigmoid. The neural network can be trained using backpropagation and gradient descent.

## Features

- Support for multiple hidden layers.
- Customizable hidden layer activation functions (`relu` or `sigmoid`).
- Customizable output layer activation functions (`sigmoid` or `relu`).
- Training using backpropagation and gradient descent.
- Visualization of training error over time.

## Installation
#### You can either pip install the package or just download the repo!
### 1. Pip install 

```powershell
pip install git+https://github.com/Rafipilot/Rafi_neural_network
```

### 2. Download repo
1. Ensure you have Python installed on your system (version 3.x is recommended).
2. Clone or download this repository.
3. Install the required libraries by running:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Initializing the Neural Network

You can initialize the neural network by providing the structure of the network (i.e., number of neurons in each layer) and specifying the activation function for the hidden and output layers.

```python
import neural_network.neural_net as neural_net


# Initialize the network
nn = neural_net.NeuralNetwork(layer_sizes=[3, 5, 1], activation='relu', output_activation='sigmoid')
```

layer_sizes: List where each value represents the number of neurons in each layer.

activation: Activation function for the hidden layers. Options: 'relu', 'sigmoid'.

output_activation: Activation function for the output layer. Options: 'sigmoid', 'relu'.

### Training the Neural Network

To train the neural network, call the train method with the training data, number of iterations, and learning rate.

```python
training_inputs = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]
training_outputs = [[0], [1], [1], [0]]

nn.train(training_inputs, training_outputs, iterations=10000, learning_rate=0.01)
```

training_inputs: List of input examples.

training_outputs: Corresponding output labels.

iterations: Number of iterations for the training process.

learning_rate: Learning rate for gradient descent.

### Making Predictions

Once the network is trained, you can use the think method to make predictions based on new inputs.

```python
new_input = [1, 0, 0]

output = nn.think(new_input)

print(f"Prediction for {new_input}: {output}")
```

## Example Usage
Here is an example of using the neural network:

```python
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
```



























