import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time  # Import the time module

# Start the timer
start_time = time.time()

# Load CIFAR-10 dataset using TensorFlow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Flatten labels
y_train = y_train.flatten()
y_test = y_test.flatten()

# Normalize the data (0 to 1)
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# Flatten images
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Standardize data so that they have mean=0 and standard deviation=1
def standardize(X, mean=None, std=None):
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
        std[std == 0] = 1  # Avoid division by zero

    return (X - mean) / std, mean, std

x_train, mean, std = standardize(x_train)
x_test, _, _ = standardize(x_test, mean, std)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=200)
x_train_pca = pca.fit_transform(x_train)  # Fit PCA on training data and transform
x_test_pca = pca.transform(x_test)        # Transform test data

print(f"Original number of features: {x_train.shape[1]}")
print(f"Reduced number of features after PCA: {x_train_pca.shape[1]}")

# Define parameter grid for hyperparameter tuning (gamma only)
param_grid = {
    'kernel': ['rbf'],  # Radial basis function kernel
    'gamma': [0.001, 0.01, 0.1, 1, 10]  # Hyperparameter for kernel coefficient
}

# Create the SVM model with C fixed at 1
svm = SVC(C=10.0)

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(x_train_pca, y_train)

# Print the best parameters and cross-validation accuracy
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.2f}")

# Evaluate the best model on the test data
best_svm = grid_search.best_estimator_
test_accuracy = best_svm.score(x_test_pca, y_test)
print(f"Test Accuracy of Best Model: {test_accuracy * 100:.2f}%")

# End the timer and calculate runtime
end_time = time.time()
runtime = end_time - start_time
print(f"Total Runtime: {runtime:.2f} seconds")
