import numpy as np
import matplotlib.pyplot as plt

X = np.vstack(([0, 0], [0, 1], [1, 0], [1, 1]))
t = np.array([0, 1, 1, 0]).reshape(-1, 1)
alpha = 1
W1 = np.random.rand(2, 16)
W2 = np.random.rand(16, 1)


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


Loss = []
for i in range(10000):
    z = sigmoid(np.dot(X, W1))
    y = sigmoid(np.dot(z, W2))
    loss = 1 / 4 * np.sum((y - t) ** 2)
    grad_W2 = 2 * (np.dot(y.T, (y - t) * y * (1 - y)))
    grad_W1 = 2 * np.dot(X.T, np.dot((y - t) * y * (1 - y), W2.T) * z * (1 - z))
    W2 = W2 - alpha * grad_W2
    W1 = W1 - alpha * grad_W1
    Loss.append(loss)

print(Loss[-1])
