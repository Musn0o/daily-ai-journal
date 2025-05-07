import numpy as np

"""NumPy Dot Product Exercises:"""

"""1.Dot product of two vectors:

    Create two 1D NumPy arrays, vector1 with values [1, 2, 3] and vector2 with values [4, 5, 6].
    Calculate their dot product using np.dot() and print the result."""

# TODO 1.Dot product of two vectors. Done!
vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])

combined_v = vector1 @ vector2
combined_v2 = vector1.dot(vector2)
""" 1*4 + 2*5 + 3*6 = 32"""
print(f"The dot product using @ == np.dot() is {combined_v} == {combined_v2}")


"""2.Dot product of a vector and a matrix:

    Create a 2D NumPy array matrix_c with the values:

    [[1, 2],
     [3, 4],
     [5, 6]]

    Create a 1D NumPy array vector3 with the values [7, 8].
    Calculate the dot product of matrix_c and vector3 using np.dot() and print the result. 
    Think about the order of the arrays in the np.dot() function – does it matter?"""

# TODO 2.Dot product of a vector and a matrix. Done!
matrix_c = np.array([[1, 2], [3, 4], [5, 6]])
vector3 = [7, 8]
""" 1*7 + 2*8 = 23 , 3*7 + 4*8 = 53 , 5*7 + 6*8 = 83 result would be [23, 53, 83]
    The order of arrays matter? of course yes 
    if matrix_c = np.array([[5, 6], [3, 4], [1, 2]])
    This would return [83, 53, 23]"""
d_product = matrix_c @ vector3
print(f"The dot product of matrix_c and vector3 {d_product}")
matrix_c = np.array([[5, 6], [3, 4], [1, 2]])
print(f"Reversed matrix_c to verify the answer {matrix_c.dot(vector3)}")


"""3.Dot product of two matrices:

    Create a 2D NumPy array matrix_d with the values:

    [[1, 0],
     [0, 1]]

    Create a 2D NumPy array matrix_e with the values:

    [[4, 5],
     [6, 7]]

    Calculate their dot product using the @ operator and print the result."""

# TODO 3.Dot product of two matrices. Done!
matrix_d = np.array([[1, 0], [0, 1]])
matrix_e = np.array([[4, 5], [6, 7]])
"""(1*4 + 0*6) = 4 , (1*5 + 0*7) = 5 , (0*4 + 1*6) = 6 , (0*5 + 1*7) = 7 >> [[4,5],[6,7]]"""
print(f"The dot product using the @ operator \n {matrix_d @ matrix_e}")


"""4.Dot product with incompatible shapes:

    Create a 1D NumPy array vector4 with the values [1, 2] and a 1D NumPy array vector5 with the values [1, 2, 3].
    Try to calculate the dot product of vector4 and vector5 using np.dot().
    What happens? (No need to solve this, just observe the output or error)."""


# TODO 4.Dot product with incompatible shapes. Done!
vector4 = np.array([1, 2])
vector5 = np.array([1, 2, 3])

try:
    v4_5 = vector4.dot(vector5)
    print(f"The dot product of vector4 and vector5 {v4_5}")
except Exception as e:
    print(e)
    gem = np.array([2])
    vector4 = np.append(vector4, gem)
    v4_5 = vector4.dot(vector5)
    """ 1*1 + 2*2 + 2*3 = 11 """
    print(f"The dot product of vector4 and vector5 {v4_5}")

"""Observation as we can see from the e Exception we cannot use 2 arrays with different shapes(dims)
   both vectors or matrics must have same shape"""
