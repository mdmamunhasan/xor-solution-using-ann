"""
XOR solution by neural network
"""

import numpy as np
from matplotlib import pyplot as plt


def sigmoid(z):
    """
    Activation function
    :param z:
    :return:
    """
    return 1 / (1 + np.exp(-z))


def initialize_parameters(n_x, n_h, n_y):
    """
    Initialize the network parameters.
    In our case, we are initializing all the weights to random numbers between 0 and 1.
    Bias terms are initialized to 0.
    Instead of passing around multiple variables, we are using a dictionary to pass all the weights and biases.
    :param n_x: int
    :param n_h: int
    :param n_y: int
    :return:
    """
    W1 = np.random.randn(n_h, n_x)  # weight
    b1 = np.zeros((n_h, 1))  # bias
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1, "b1": b1,
        "W2": W2, "b2": b2
    }

    return parameters


def forward_propagation(X, Y, parameters):
    """
    Forward propagation  just involves multiplication of weights with inputs, addition with biases and sigmoid function.
    :param X:
    :param Y:
    :param parameters:
    :return:
    """
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2)
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    cost = -np.sum(logprobs) / m

    return cost, cache, A2


def backward_propagation(X, Y, cache):
    """
    Backward propagation that we want to make small changes to the weights and biases
    so that our network output moves towards the expected output.
    :param X:
    :param Y:
    :param cache:
    :return:
    """
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2) = cache

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, A1 * (1 - A1))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    gradients = {
        "dZ2": dZ2, "dW2": dW2, "db2": db2,
        "dZ1": dZ1, "dW1": dW1, "db1": db1
    }

    return gradients


def update_parameters(parameters, grads, learning_rate):
    """
    We are updating the weights based on the negative gradients.
    :param parameters:
    :param grads:
    :param learning_rate:
    :return:
    """
    parameters["W1"] = parameters["W1"] - learning_rate * grads["dW1"]
    parameters["W2"] = parameters["W2"] - learning_rate * grads["dW2"]
    parameters["b1"] = parameters["b1"] - learning_rate * grads["db1"]
    parameters["b2"] = parameters["b2"] - learning_rate * grads["db2"]
    return parameters


def evaluate_performance(X, Y, parameters, losses, plot=False):
    cost, _, A2 = forward_propagation(X, Y, parameters)
    pred = (A2 > 0.5) * 1.0
    print("Activation Output", A2)
    print("Final Prediction", pred)
    if plot:
        plt.figure()
        plt.plot(losses)
        plt.show()


def main():
    """
    Put together the complete neural network model to learn the XOR truth table
    :return:
    """
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    # Y = np.array([[0, 0, 0, 1]])  # AND
    # Y = np.array([[0, 1, 1, 1]])  # OR
    # Y = np.array([[1, 1, 1, 0]])  # NAND
    Y = np.array([[0, 1, 1, 0]])  # XOR
    n_h = 2
    n_x = X.shape[0]
    n_y = Y.shape[0]
    parameters = initialize_parameters(n_x, n_h, n_y)
    num_iterations = 100000
    learning_rate = 0.01
    losses = np.zeros((num_iterations, 1))

    print("Expected Output", Y)

    print("Performance before training:")
    evaluate_performance(X, Y, parameters, losses)

    print("Train the Network...")
    for i in range(num_iterations):
        losses[i, 0], cache, A2 = forward_propagation(X, Y, parameters)
        grads = backward_propagation(X, Y, cache)
        parameters = update_parameters(parameters, grads, learning_rate)
    print(f"Training completed with {num_iterations} epochs")

    print("Performance after training:")
    evaluate_performance(X, Y, parameters, losses)


if __name__ == '__main__':
    losses = main()
