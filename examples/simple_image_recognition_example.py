import numpy as np
import streamlit as st
import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_net import NeuralNetwork

st.set_page_config(page_title="Demo", layout="wide")

st.title("Simple Image Recognition")

# Training dataset: 2x2 images flattened to 4 inputs
training_inputs = np.array([[1, 1, 0, 0],  # Horizontal line
                        [1, 0, 1, 0],  # Vertical line
                        [1, 0, 0, 1],  # Diagonal line
                        [0, 0, 0, 0]]) # No pattern

# Horizontal, Vertical, Diagonal, No pattern
training_outputs = np.array([[1, 0, 0, 0],  # Horizontal line
                            [0, 1, 0, 0],  # Vertical line
                            [0, 0, 1, 0],  # Diagonal line
                            [0, 0, 0, 1]]) # No pattern

q = 4
if "neural_network" not in st.session_state:
    st.session_state.neural_network = NeuralNetwork(training_inputs, training_outputs, q)


st.session_state.neural_network.train(training_inputs, training_outputs, 20000)

# Create a more user-friendly 2x2 grid using checkboxes
st.write("Draw the input image by checking the boxes below:")

# Initialize state for binary input
if 'grid' not in st.session_state:
    st.session_state.grid = [0, 0, 0, 0]

def update_grid(index):
    st.session_state.grid[index] = 1 - st.session_state.grid[index]

# Display checkboxes in a 2x2 grid
col1, col2 = st.columns(2)
with col1:
    st.checkbox(' ', value=bool(st.session_state.grid[0]), key='checkbox_0', on_change=lambda: update_grid(0))
    st.checkbox(' ', value=bool(st.session_state.grid[2]), key='checkbox_2', on_change=lambda: update_grid(2))
with col2:
    st.checkbox(' ', value=bool(st.session_state.grid[1]), key='checkbox_1', on_change=lambda: update_grid(1))
    st.checkbox(' ', value=bool(st.session_state.grid[3]), key='checkbox_3', on_change=lambda: update_grid(3))

if st.button("Classify Image"):
    binary_input = np.array(st.session_state.grid).reshape(1, 4)
    print("Considering new situation:", binary_input)
    output = st.session_state.neural_network.think(binary_input)
    st.write("Output:", output)
