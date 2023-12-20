import numpy as np

def generate_invertible_matrix(n):
    A = np.random.randint(1, 10, size=(n, n))

    while np.linalg.det(A) == 0:
        A = np.random.randint(1, 10, size=(n, n))

    return A

def generate_invertible_symmetric_matrix(n):
    B = np.random.randint(1, 10, size=(n, n))
    A = B + B.T
    while np.linalg.det(A) == 0:
        B = np.random.randint(1, 10, size=(n, n))
        A = B + B.T

    return A

def eigen_decomposition_and_reconstruction(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    A_reconstructed = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
    return A_reconstructed

def check_reconstruction(A, A_reconstructed):
    reconstruction_error = np.linalg.norm(A - A_reconstructed)

    return reconstruction_error

n = int(input("Enter the dimension of the matrix (n): "))

A = generate_invertible_symmetric_matrix(n)

A_reconstructed = eigen_decomposition_and_reconstruction(A)

reconstruction_error = check_reconstruction(A, A_reconstructed)

print("\nOriginal Matrix A:\n", A)
print("\nReconstructed Matrix A:\n", A_reconstructed)
print("\nReconstruction Error:", reconstruction_error)

if np.allclose(A, A_reconstructed):
    print("\nReconstruction is accurate.")
else:
    print("\nReconstruction is not accurate.")
