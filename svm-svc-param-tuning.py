import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import time

# Start the timer
start_time = time.time()

# Load CIFAR-10 dataset using TensorFlow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Flatten labels
y_train = y_train.flatten()
y_test = y_test.flatten()

# Filter for airplane (0) and automobile (1) classes
airplane_automobile_train_indices = np.where((y_train == 0) | (y_train == 1))[0]
airplane_automobile_test_indices = np.where((y_test == 0) | (y_test == 1))[0]

x_train = x_train[airplane_automobile_train_indices]
y_train = y_train[airplane_automobile_train_indices]
x_test = x_test[airplane_automobile_test_indices]
y_test = y_test[airplane_automobile_test_indices]

# Normalize the data (0 to 1)
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# Flatten images
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Standardize data using StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=200)
x_train_pca = pca.fit_transform(x_train)  # Fit PCA on training data and transform
x_test_pca = pca.transform(x_test)        # Transform test data

print(f"Original number of features: {x_train.shape[1]}")
print(f"Reduced number of features after PCA: {x_train_pca.shape[1]}")


# Define parameter grid for hyperparameter tuning (C and coef0 for linear kernel)
param_grid = {
    'kernel': ['linear'],  # Linear kernel
    'C': [0.1, 1.0, 10, 100],  # Regularization parameter
    'coef0': [0, 1, 10]  # Bias term, though it's less important for linear kernel
}

# Create the SVM model
svm = SVC()

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(x_train_pca, y_train)

# Print the best parameters and accuracy
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.2f}")

# Evaluate the best model on the test data
best_svm = grid_search.best_estimator_
test_accuracy = best_svm.score(x_test_pca, y_test)
print(f"Test Accuracy of Best Model: {test_accuracy * 100:.2f}%")

# End the timer and print runtime
end_time = time.time()
print(f"Runtime: {end_time - start_time:.2f} seconds")
