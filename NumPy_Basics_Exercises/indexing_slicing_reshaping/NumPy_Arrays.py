# 📌 Task 1: Create NumPy Arrays

# Importing the NumPy library with an alias 'np'
import numpy as np

# ToDO 1 Create a 1D array of numbers from 1 to 10.

# Creating an array that must include 1 & 10 for the ToDo
one_d_array = np.arange(1, 11)
# Printing the array from 0 to 10
print("Array from 1 to 10:")
print(f"{one_d_array}\n")

# ToDO 2 Create a 2D array (3x3) filled with random numbers between 1 and 100.
# Creating 2D array (3 x 3) that must include rand int from 1 to 100
two_d_array = np.random.randint(1, 101, size=(3, 3))
# Printing the 2D (3 x 3) array from 1 to 100
print("Array with random numbers from 1 to 100:")
print(f"{two_d_array}\n")

# ToDO 3 Create an array of 10 zeros and an array of 10 ones.

# Creating an array of 10 zeros using np.zeros()
zeros = np.zeros(10, dtype=np.float64)
# Printing a message indicating an array of 10 zeros
print("An array of 10 zeros:")
# Printing the array of 10 zeros
print(zeros)

# Creating an array of 10 ones using np.ones()
ones = np.ones(10, dtype=np.float64)
# Printing a message indicating an array of 10 ones
print("An array of 10 ones:")
# Printing the array of 10 ones
print(f"{ones}\n")

# 📌 Task 2: Indexing & Slicing

# ToDo 1 Extract only even numbers from your 1D array.
# even values in the array eve
print(f"Array with even Numbers: {one_d_array[one_d_array % 2 == 0]}\n")

# ToDo 2 Select the second row from your 2D array.
print(f"Second Row is {two_d_array[1]}\n")

# ToDo 3 Reverse the 1D array.
print(f"Reversed Array {one_d_array[::-1]}\n")

# 📌 Task 3: Mathematical Operations

# ToDo 1 Multiply all elements in the 2D array by 2.
print(f"Multiplied 2D array by 2:\n {two_d_array * 2}\n")

# ToDo 2 Compute the mean, max, and min of the 2D array.
print(
    f"Mean is : {two_d_array.mean()} \n Max is : {two_d_array.max()} \n Min is : {two_d_array.min()}\n"
)

# ToDo 3 Normalize the 2D array (scale values between 0 and 1).
# Perform min-max normalization by subtracting min value from the array / max value - min value
min_val = two_d_array.min()
max_val = two_d_array.max()
if max_val != min_val:
    my_normalized_2d = (two_d_array - min_val) / (max_val - min_val)
else:
    my_normalized_2d = np.zeros_like(two_d_array)  # Avoid division by zero
print(f"Normalized Array between 0 & 1:\n {my_normalized_2d}")
