import numpy as np
from sklearn.datasets import fetch_openml


def get_mnist():
    """ 
    Fetches the MNIST dataset and prepares it for use in our neural network.
    Images are normalized so pixel values are [0.0, 1.0] and so labels are 
    integers between [0,9].
    
    Returns:
        (np.ndarray, np.ndarray): Images and labels.
    """
        
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    
    X = X / 255.0 # Normalize images

    y = np.array([int(i) for i in y])

    return X, y