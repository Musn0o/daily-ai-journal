import numpy as np

"""NumPy Broadcasting Exercises:"""

"""1.Scalar and Array:

    Create a NumPy array array_ex1 with values [10, 20, 30, 40].
    Subtract the scalar value 5 from each element of array_ex1 using broadcasting and print the result."""

# TODO 1.Scalar and Array. Done!

array_ex1 = np.array([10, 20, 30, 40])
val = 5

result = array_ex1 - val

print(f"The substraction result is >> {result}")


"""2.1D Array and 2D Array (Row-wise Broadcasting):

    Create a 2D NumPy array array_ex2 with the values:

    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

    Create a 1D NumPy array array_ex2_row with the values [10, 20, 30].
    Add array_ex2_row to each row of array_ex2 using broadcasting and print the result."""

# TODO 2. 1D Array and 2D Array (Row-wise Broadcasting). Done!

array_ex2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
array_ex2_row = np.array([10, 20, 30])

result_2 = array_ex2 + array_ex2_row

print(f"Addition of array_ex2_row to each row of array_ex2 \n {result_2}")


"""3.1D Array and 2D Array (Column-wise Broadcasting):

    Create the same 2D NumPy array array_ex3 as in the previous exercise:

    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

    Create a 1D NumPy array array_ex3_col with the values [100, 200, 300].
    To broadcast this array as a column to array_ex3, you'll need to reshape it. Reshape array_ex3_col to be a column vector (shape (3, 1)).
    Add the reshaped array_ex3_col to each column of array_ex3 using broadcasting and print the result."""

# TODO 3. 1D Array and 2D Array (Column-wise Broadcasting). Done!

array_ex3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
array_ex3_col = np.array([100, 200, 300])
print(f"unshaped {array_ex3_col.shape}")

reshaped = array_ex3_col.reshape(3, 1)
print(f"reshaped {reshaped.shape}")
result_3 = array_ex3 + reshaped
print(
    f"Addition of the reshaped array_ex3_col to each column of array_ex3 \n {result_3}"
)


"""4.Incompatible Shapes:

    Create a NumPy array array_ex4_a with shape (2, 3) and some values.
    Create a NumPy array array_ex4_b with shape (3, 2) and some values.
    Try to perform an element-wise addition of array_ex4_a and array_ex4_b. What happens? 
    (No need to solve, just observe the output or error and explain why it occurs based on the broadcasting rules)."""

# TODO 4.Incompatible Shapes. Done!

array_ex4_a = np.array([[1, 2, 3], [4, 5, 6]])
array_ex4_b = np.array([[10, 20], [30, 40], [50, 60]])

try:
    result_4 = array_ex4_a + array_ex4_b
    print(result_4)
except Exception as e:
    print(f"{e} So we will reshape array_ex4_a to (3, 2)")

    reshaped_array_ex4_a = array_ex4_a.reshape(3, 2)
    result_4 = reshaped_array_ex4_a + array_ex4_b
    print(
        f"Now reshaped_array_ex4_a \n {reshaped_array_ex4_a} \n + \n {array_ex4_b} \n = \n {result_4}"
    )

"""result_4 operation won't work because the shape is different and none of them got a value 1 to stretch"""
