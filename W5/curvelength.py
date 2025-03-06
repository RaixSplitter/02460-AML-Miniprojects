import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()


def curve_length(coords: np.array):
    """
    Calculate the length of a curve defined by the given coordinates, by adding linesegments.
    
    Args: 
        coords: A 2D numpy array of shape (n, 2) where n is the number of points.
    return: 
        The approximate length of the curve.
    """
    
    n = coords.shape[0]
        
    length = 0
    
    for i in range(n - 1):
        x1, y1 = coords[i] # x1, y1 is the i-th point
        x2, y2 = coords[i + 1] # x2, y2 is the (i+1)-th point
        length += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) # Euclidean distance
    return length

def polynomial_function(constants: np.array):
    """
    Create a polynomial function from the given constants.
    
    Args: 
        constants: A 1D numpy array of shape (n) where n is the number of constants.
    return: 
        A function that calculates the value of the polynomial at a given x.
    """
    
    p = np.polynomial.Polynomial(constants)
    
    return p
    
    

if __name__ == "__main__":
    # Example usage
    function = polynomial_function([1, 3, -2, 1, -13])  # f(x) = x
    x = np.linspace(-10, 10, 25)
    y = function(x)
    coords = np.array([x, y]).T
    length = curve_length(coords)

    # Plot the curve
    plt.plot(coords[:, 0], coords[:, 1], marker='o')
    plt.show()