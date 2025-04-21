import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import random

# Load CIFAR-10 dataset from TensorFlow
cifar10 = tf.keras.datasets.cifar10
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Normalize and reshape data
X_train = X_train.reshape(X_train.shape[0], -1).T / 255.0  # Flatten and normalize
X_test = X_test.reshape(X_test.shape[0], -1).T / 255.0     # Flatten and normalize

Y_train = Y_train.flatten()  # Flatten labels to 1D array
Y_test = Y_test.flatten()    # Flatten labels to 1D array

# Debugging shape information
print(f"Training set shape: {X_train.shape}, Labels: {Y_train.shape}")
print(f"Test set shape: {X_test.shape}, Labels: {Y_test.shape}")

def pca(X, num_components):
    # Step 1: Center the data
    mean_vector = np.mean(X, axis=1, keepdims=True)  # Compute mean for each feature
    X_centered = X - mean_vector  # Center the data

    # Step 2: Compute covariance matrix
    covariance_matrix = np.cov(X_centered)

    # Step 3: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Step 4: Sort eigenvectors by descending eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors[:, sorted_indices[:num_components]]  # Select top components

    # Step 5: Project data onto the top principal components
    X_reduced = top_eigenvectors.T @ X_centered  # Transform data

    return X_reduced, top_eigenvectors, mean_vector

# Apply PCA to reduce dimensions to 200
num_components = 200
X_train_pca, pca_components, train_mean_vector = pca(X_train, num_components=num_components)

# Center the test data using the mean of the training data
X_test_centered = X_test - train_mean_vector  # Center test data
X_test_pca = pca_components.T @ X_test_centered  # Project the test data

# Debugging shape information
print(f"Training set shape after PCA: {X_train_pca.shape}, Labels: {Y_train.shape}")
print(f"Test set shape after PCA: {X_test_pca.shape}, Labels: {Y_test.shape}")

# Neural network functions
def init_params(input_size=200):
    W1 = np.random.rand(50, input_size) - 0.5 
    b1 = np.random.rand(50, 1) - 0.5
    W2 = np.random.rand(10, 50) - 0.5   # Hidden layer size set to 50
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    return Z > 0

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    return Z1, A1, Z2, None  # A2 is unused with hinge loss

def compute_hinge_loss(Z2, Y):
    """
    Compute the multi-class hinge loss.
    """
    m = Y.size
    correct_class_scores = Z2[Y, np.arange(m)]
    margins = np.maximum(0, 1 + Z2 - correct_class_scores)
    margins[Y, np.arange(m)] = 0  # Ignore correct class
    loss = np.sum(margins) / m
    return loss

def backward_prop_hinge(Z1, A1, Z2, W1, W2, X, Y):
    """
    Backpropagation for hinge loss.
    """
    m = X.shape[1]
    margins = 1 + Z2 - Z2[Y, np.arange(m)]
    margins[Y, np.arange(m)] = 0
    binary = margins > 0

    dZ2 = binary.astype(float)
    dZ2[Y, np.arange(m)] -= np.sum(binary, axis=0)

    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

def get_predictions(Z2):
    return np.argmax(Z2, axis=0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def create_mini_batches(X, Y, batch_size):
    m = X.shape[1]
    indices = np.random.permutation(m)
    X_shuffled = X[:, indices]
    Y_shuffled = Y[indices]
    mini_batches = []
    for k in range(0, m, batch_size):
        X_batch = X_shuffled[:, k:k + batch_size]
        Y_batch = Y_shuffled[k:k + batch_size]
        mini_batches.append((X_batch, Y_batch))
    return mini_batches

def gradient_descent(X_train_pca, Y_train, X_test_pca, Y_test, alpha, epochs, batch_size):
    W1, b1, W2, b2 = init_params(input_size=X_train_pca.shape[0])
    train_losses = []
    test_accuracies = []
    train_accuracies = []

    for i in range(epochs):
        mini_batches = create_mini_batches(X_train_pca, Y_train, batch_size)
        epoch_loss = 0
        for X_batch, Y_batch in mini_batches:
            Z1, A1, Z2, _ = forward_prop(W1, b1, W2, b2, X_batch)
            loss = compute_hinge_loss(Z2, Y_batch)
            epoch_loss += loss
            dW1, db1, dW2, db2 = backward_prop_hinge(Z1, A1, Z2, W1, W2, X_batch, Y_batch)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        epoch_loss /= len(mini_batches)
        train_losses.append(epoch_loss)

        # Calculate training accuracy
        _, _, Z2_train, _ = forward_prop(W1, b1, W2, b2, X_train_pca)
        train_predictions = get_predictions(Z2_train)
        train_accuracy = get_accuracy(train_predictions, Y_train)
        train_accuracies.append(train_accuracy)

        # Calculate test accuracy
        _, _, Z2_test, _ = forward_prop(W1, b1, W2, b2, X_test_pca)
        test_predictions = get_predictions(Z2_test)
        test_accuracy = get_accuracy(test_predictions, Y_test)
        test_accuracies.append(test_accuracy)

        if (i + 1) % 20 == 0 or i == epochs - 1:
            print(f"Epoch {i + 1}/{epochs}: Loss = {epoch_loss:.4f}, Train Accuracy = {train_accuracy:.4f}, Test Accuracy = {test_accuracy:.4f}")

    return W1, b1, W2, b2, test_predictions

# Train the neural network
W1, b1, W2, b2, test_predictions = gradient_descent(X_train_pca, Y_train, X_test_pca, Y_test, alpha=0.01, epochs=300, batch_size=64)


