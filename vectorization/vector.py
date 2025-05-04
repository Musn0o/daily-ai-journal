from random import randint
import numpy as np

"""NumPy Vectorization Exercises:"""

"""1.Add a Constant:

    Create a large list of numbers (e.g., using range(100000)).
    Create a NumPy array from that list.
    Problem: Add the value 10 to each element of the list/array.
    Write the vectorized NumPy code to achieve this."""

list_a = list(range(100000))
num_arr = np.array(list_a)
list_sum = [n + 10 for n in list_a]
vectorized = num_arr + 10
# print(f"Output with added value of 10 to each element of the list {list_sum}")
# print(f"Output with added value of 10 to each element of the array {vectorized}")


"""2.Element-wise Multiplication:

    Create two lists of the same size (e.g., 50000 random integers).
    Create two NumPy arrays from these lists.
    Problem: Multiply each element in the first list/array by the corresponding element in the second list/array.
    Write the vectorized NumPy code for this.
"""
size = randint(0, 50000)
list_b = list(range(size))
list_c = list(range(size))

num_arr_a = np.array(list_b)
num_arr_b = np.array(list_c)

# multiplied_lists = [n * i for n in list_b for i in list_c]
multiplied = np.multiply(num_arr_a, num_arr_b)
# print(f"Multiplied lists {multiplied_lists}")
# print(f"Multiplied arrays {multiplied}")


"""3.Thresholding and Summation:

    Create a NumPy array of 100000 random numbers between 0 and 100.
    Problem: Find the sum of all elements in the array that are greater than 50.
    Write the vectorized NumPy code to achieve this efficiently (hint: think about boolean indexing!).
"""
rnd_array = np.random.randint(100, size=100000)
print(rnd_array)
print(f"Sum of random array {np.sum(rnd_array)}")
boolean_array = [n > 50 for n in np.nditer(rnd_array)]

filtered_elements = rnd_array[boolean_array]
print(filtered_elements)
print(f"Sum of filtered array {np.sum(filtered_elements)}")


"""4.Applying a Mathematical Function:

    Create a NumPy array of 50000 numbers (e.g., using np.linspace(0, 2*np.pi, 50000)).
    Problem: Calculate the sine of each element in the array.
    Write the vectorized NumPy code using NumPy's mathematical functions.
    """

math_arr = np.linspace(0, 2 * np.pi, 50000)
print(math_arr)
result = np.sin(math_arr)
print(result)
