import numpy as np
import matplotlib.pyplot as plt

def compute_covariance(data1 : np.array, data2 : np.array, normalize : bool) -> float:
    """
    Compute the covariance between two datasets and normalize it by the number of samples

    Parameters:
    data1 (np.array): first dataset
    data2 (np.array): second dataset
    normalize (bool): whether to normalize the covariance

    Returns:
    covariance (float): covariance between the two datasets
    """
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    cov = np.mean((data1 - mean1) * (data2 - mean2))
    if not normalize : return cov
    else : 
        factor = np.sqrt(np.mean((data1-mean1)**2) * np.mean((data2-mean2)**2))
        return cov / factor


def linear_classifier(x: np.ndarray, w: np.ndarray, b: float):
    """
    Compute the prediction of a linear classifier

    Parameters:
    x (np.ndarray): data points to predict
    w (np.ndarray): weights of the linear classifier
    b (float): bias of the linear classifier

    Returns:
    prediction (np.ndarray): predictions of the linear classifier
    """
    for x_data in x:
        assert x_data.shape == w.shape, "Input x and weights w must have the same dimensions"
    return np.sign(np.dot(x, w) + b)



def plot_decision_boundary(w, b, X=None, y=None, x_range=(-10, 10), y_range=(-10, 10), n_points=100):
    """
    Plot the decision boundary of a linear classifier

    Parameters:
    w (np.ndarray): weights of the linear classifier
    b (float): bias of the linear classifier
    X (np.ndarray): data points to plot the decision boundary
    y (np.ndarray): labels of the data points
    x_range (tuple): range of the x-axis
    y_range (tuple): range of the y-axis
    n_points (int): number of points to plot the decision boundary
    """
    # Check if X is provided and has correct dimensions
    if X is None:
        X = np.array([[x_range[0], y_range[0]], [x_range[1], y_range[1]]])
    else:
        if X.shape[1] != 2:
            raise ValueError("X must have 2 features")

    # Create a grid of points
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x1, x2 = np.meshgrid(np.linspace(x1_min, x1_max, n_points),
                         np.linspace(x2_min, x2_max, n_points))

    # Calculate the predictions
    grid = np.c_[x1.ravel(), x2.ravel()]
    predictions = linear_classifier(grid, w, b).reshape(x1.shape)

    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(x1, x2, predictions, alpha=0.8, cmap='RdYlBu')

    # Plot training data if provided
    if X is not None and y is not None:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='black')

    plt.colorbar()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Decision Boundary')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.show()


def perceptron_without_offset(Xs, ys , epochs):
    """
    Train a perceptron without offset using the perceptron algorithm

    Parameters:
    Xs (np.ndarray): data points
    ys (np.ndarray): labels
    epochs (int): number of epochs

    Returns:
    ws (np.ndarray): weights
    training_errors (list): training errors
    """
    # initialize weights
    ws = np.zeros_like(Xs[0]) # make sure ws is the same shape as Xs[0]

    training_errors = []
    for epoch in range(epochs):
        errors = 0
        for x, y in zip(Xs, ys):
            prediction = linear_classifier(x, ws, 0)
            if prediction != y:
                ws += y * x
                errors += 1
            else:
                pass
        # calculate training error
        training_error = errors / len(Xs)
        training_errors.append(training_error)
        # print(f"Epoch {epoch}, Training Error: {training_error}")
    return ws, training_errors

def averaged_perceptron_with_offset(Xs, ys, epochs):
    """
    Train a averaged perceptron with offset using the averaged perceptron algorithm

    Parameters:
    Xs (np.ndarray): data points
    ys (np.ndarray): labels
    epochs (int): number of epochs

    Returns:
    ws (np.ndarray): weights
    b (float): bias
    training_errors (list): training errors
    """
    # initialize weights
    ws = np.zeros_like(Xs[0]) # make sure ws is the same shape as Xs[0]
    b = 0
    c = 1
    
    training_errors = []
    for epoch in range(epochs):
        errors = 0
        for x, y in zip(Xs, ys):
            prediction = linear_classifier(x, ws, b)
            if prediction != y:
                errors += 1
                ws += c*y * x
                b += c*y
            c -= 1/(len(Xs)*epochs)
        # calculate training error
        training_error = errors / len(Xs)
        training_errors.append(training_error)
    return ws, b, training_errors