import numpy as np

"""NumPy Transpose Exercises:"""

"""1.Transpose a 2x3 matrix:
    Create a 2D NumPy array matrix_a with the following values:

    [[1, 2, 3],
    [4, 5, 6]]

    Find the transpose of matrix_a and print it."""

# ToDO 1.Transpose a 2x3 matrix. Done!
matrix_a = np.array([[1, 2, 3], [4, 5, 6]])
print(f"2D NumPy 2x3 array \n{matrix_a}\n it's shape is >> {matrix_a.shape}")

transposed_a = matrix_a.T
print(
    f"Transposed of matrix_a \n{transposed_a}\n it's shape is >> {transposed_a.shape}"
)


"""2.Transpose a 3x2 matrix:

    Create a 2D NumPy array matrix_b with the following values:

    [[10, 20],
     [30, 40],
     [50, 60]]

    Find the transpose of matrix_b and print it."""

# ToDO 2.Transpose a 3x2 matrix. Done!
matrix_b = np.array([[10, 20], [30, 40], [50, 60]])
print(f"2D NumPy 3x2 array \n{matrix_b}\n it's shape is >> {matrix_b.shape}")

transposed_b = matrix_b.T
print(
    f"Transposed of matrix_b \n{transposed_b}\n it's shape is >> {transposed_b.shape}"
)


"""3.Transpose a 1D array:

    Create a 1D NumPy array array_1d with the values [7, 8, 9].
    Find the transpose of array_1d and print it. 
    Observe the shape of the original and the transposed array. What do you notice?"""

# ToDO 3.Transpose a 1D array. Done!

array_1d = np.array([7, 8, 9])
print(f"1D NumPy array of array_1d \n{array_1d}\n it's shape is >> {array_1d.shape}")

transposed_1d = array_1d.T
print(
    f"Transposed of array_1d \n{transposed_1d}\n it's shape is >> {transposed_1d.shape}"
)

transposed_1d_fun = array_1d.transpose(
    0,
)
print(
    f"Transposed of array_1d with transpose() \n{transposed_1d_fun}\n it's shape is >> {transposed_1d_fun.shape}"
)
"""Transpose wont't work on the 1D array because it got only 1 axis what it's gonna swap it with? nothing"""


"""4.Transpose a square matrix:

    Create a 2D NumPy array square_matrix with the following values:

    [[1, 4],
     [2, 5]]

    Find the transpose of square_matrix and print it. What do you notice about the transpose of this specific matrix?"""

# ToDO 4.Transpose a square matrix. Done!

square_matrix = np.array([[1, 4], [2, 5]])
print(
    f"2D NumPy array square_matrix \n{square_matrix}\n it's shape is >> {square_matrix.shape}"
)

transposed_square_matrix = square_matrix.T
print(
    f"Transposed 2D NumPy array square_matrix \n{transposed_square_matrix}\n it's shape is >> {transposed_square_matrix.shape}"
)

"""As I expected the numbers swapped but the shape is the same"""
