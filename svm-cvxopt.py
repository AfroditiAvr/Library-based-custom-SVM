import numpy as np
import cvxopt
import tensorflow as tf
from sklearn.decomposition import PCA

class SVM_classifier():

    def __init__(self, c_parameter):
        self.c_parameter = c_parameter  # Using C instead of lambda
    
    def fit(self, X, Y):        
        self.m, self.n = X.shape
        self.X = X
        self.Y = Y

        print("Starting training...")
        
        # Compute the kernel matrix (linear kernel here)
        K = np.dot(self.X, self.X.T)

        # Define the matrices for quadratic programming (dual problem)
        self.Y = np.where(self.Y == 0, -1, 1)  # Mapping 0 to -1 and 1 to 1
        P = np.outer(self.Y, self.Y) * K  # Quadratic matrix: Pij​=yi​yj​Kij​
        q = -np.ones(self.m)  # Linear term: a= -∑ ​αi​

        # Inequality constraints
        G = np.vstack((np.eye(self.m) * -1, np.eye(self.m)))     # Diagonal matrix of size m×m with −1 on the diagonal
        h = np.hstack((np.zeros(self.m), np.ones(self.m) * self.c_parameter))

        # Equality constraints
        A = self.Y.reshape(1, -1)  # A=[y1 ​​y2​...​ym​​] (labels)
        b = np.zeros(1)  # Scalar zero

        # prepare the matrices for the cvxopt quadratic programming solver
        P = cvxopt.matrix(P)
        q = cvxopt.matrix(q)
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h)
        A = cvxopt.matrix(A, (1, self.m), 'd')  # Reshape to (1, m) and set type to 'd'
        b = cvxopt.matrix(b)

        # Solve the quadratic programming problem to find Lagrange multipliers α
        cvxopt.solvers.options['maxiters'] = 100  # Maximum number of iterations
        cvxopt.solvers.options['abstol'] = 1e-6  # Absolute tolerance
        cvxopt.solvers.options['reltol'] = 1e-6  # Relative tolerance
        cvxopt.solvers.options['feastol'] = 1e-6  # Feasibility tolerance
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alpha = np.array(solution['x']).flatten()      #solution['x'] is a matrix that holds the values of the Lagrange multipliers α

        # Compute the weight vector w= ∑ ​αi​yixi
        self.w = np.sum(alpha[:, np.newaxis] * self.Y[:, np.newaxis] * self.X, axis=0)

        # Compute the bias b (using support vectors, alpha > 0)
        support_vectors = alpha > 1e-5
        self.b = np.mean(self.Y[support_vectors] - np.dot(self.X[support_vectors], self.w))    # compute b using the support vectors (αi​>0):b=1/|S|*∑i∈S(yi−⟨w,xi⟩)

        print("Training completed.")

    def predict(self, X):
        output = np.dot(X, self.w) - self.b
        predicted_labels = np.sign(output)
        predicted_labels = np.where(predicted_labels <= -1, 0, 1)  # Map -1 to 0, 1 to 1
        return predicted_labels

# Load CIFAR-10 dataset using TensorFlow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Flatten labels
y_train = y_train.flatten()
y_test = y_test.flatten()

airplane_automobile_train_indices = np.where((y_train == 0) | (y_train == 1))[0]
airplane_automobile_test_indices = np.where((y_test == 0) | (y_test == 1))[0]

x_train = x_train[airplane_automobile_train_indices]
y_train = y_train[airplane_automobile_train_indices]
x_test = x_test[airplane_automobile_test_indices]
y_test = y_test[airplane_automobile_test_indices]

# label: airplane (0) -> 0, automobile (1) -> 1

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


pca = PCA(n_components=200)
x_train_pca = pca.fit_transform(x_train)  # Fit PCA on training data and transform
x_test_pca = pca.transform(x_test)        # Transform test data

print(f"Original number of features: {x_train.shape[1]}")
print(f"Reduced number of features after PCA: {x_train_pca.shape[1]}")


# Train the SVM
svm = SVM_classifier(c_parameter=1)
svm.fit(x_train, y_train)

# Evaluate the SVM
train_accuracy = np.mean(svm.predict(x_train) == y_train)
test_accuracy = np.mean(svm.predict(x_test) == y_test)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
