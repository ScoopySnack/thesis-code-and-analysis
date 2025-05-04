import numpy as np

def perron_frobenius(a):
    """
    Computes the Perron-Frobenius eigenvalue and eigenvector of a non-negative matrix A.

    Parameters:
    a (numpy.ndarray): A non-negative square matrix.

    Returns:
    tuple: The Perron-Frobenius eigenvalue and the corresponding eigenvector.
    """
    # Check if A is a square matrix
    if a.shape[0] != a.shape[1]: #shape[0] is the number of rows, shape[1] is the number of columns
        raise ValueError("Matrix must be square.")

    # Check if A is non-negative
    if np.any(a < 0): #any() returns True if any of the elements in the array are non-negative
        raise ValueError("Matrix must be non-negative.")

    # Compute the dominant eigenvalue and eigenvector
    eigenvalues, eigenvectors = np.linalg.eig(a) #linalg.eig() computes the eigenvalues and right eigenvectors of a square array

    # Get the index of the dominant eigenvalue
    dominant_index = np.argmax(np.abs(eigenvalues))
    #argmax() returns the indices of the maximum values along an axis and abs() returns the absolute value of each element in the array

    # Get the dominant eigenvalue and corresponding eigenvector
    dominant_eigenvalue = np.real(eigenvalues[dominant_index]) #real() returns the real part of the complex number
    #eigenvalues[dominant_index] returns the dominant eigenvalue
    #eigenvectors[:, dominant_index] returns the corresponding eigenvector
    dominant_eigenvector = np.real(eigenvectors[:, dominant_index])

    # Normalize the eigenvector
    #normalization is done by dividing each element of the eigenvector by the sum of all elements
    # This ensures that the sum of the elements of the eigenvector is 1
    dominant_eigenvector /= np.sum(dominant_eigenvector)

    return dominant_eigenvalue, dominant_eigenvector



# Example usage
if __name__ == "__main__":
    C = np.array([[1, 1, 1],
                  [1, 2, 1],
                  [1, 1, 3]])

    eigenvalue, eigenvector = perron_frobenius(C)
    print("Perron-Frobenius Eigenvalue:", eigenvalue)
    print("Perron-Frobenius Eigenvector:", eigenvector)

    # second way to compute the dominant eigenvalue and eigenvector
    # using numpy's built-in function
    # This method is not guaranteed to return the Perron-Frobenius eigenvalue, but it doesn't check for non-negativity
    A = np.array([[1, 1, 1],
                  [1, 2, 1],
                  [1, 1, 3]])

    # Compute all eigenvalues
    eigenvalues1 = np.linalg.eigvals(A)

    # Perron-Frobenius eigenvalue = maximum real part eigenvalue
    pf_eigenvalue = max(eigenvalues1.real)

    print("Perron-Frobenius eigenvalue:", pf_eigenvalue)