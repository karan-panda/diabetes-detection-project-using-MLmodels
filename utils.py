import numpy as np
import matplotlib.pyplot as plt

def plot_relu():
    # Generate x values
    x = np.linspace(-10, 10, 100)

    # Compute ReLU outputs
    relu_output = np.maximum(0, x)

    # Plot ReLU curve
    plt.figure(figsize=(8, 4))
    plt.plot(x, relu_output, label='ReLU')
    plt.title('ReLU Activation')
    plt.xlabel('x')
    plt.ylabel('ReLU(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_sigmoid():
    # Generate x values
    x = np.linspace(-10, 10, 100)

    # Compute sigmoid outputs
    sigmoid_output = 1 / (1 + np.exp(-x))

    # Plot sigmoid curve
    plt.figure(figsize=(8, 4))
    plt.plot(x, sigmoid_output, label='Sigmoid')
    plt.title('Sigmoid Activation')
    plt.xlabel('x')
    plt.ylabel('sigmoid(x)')
    plt.legend()
    plt.grid(True)
    plt.show()
