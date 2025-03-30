import numpy as np


"""NumPy Indexing, Slicing, and Reshaping Exercises:"""

"""1.Basic Indexing:
    Create a 1D NumPy array with the values [10, 20, 30, 40, 50].
    Access the first element of the array.
    Access the last element of the array.
    Create a 2D NumPy array with the shape (3, 3) containing the numbers from 1 to 9.
    Access the element in the second row and third column."""


# ToDo 1 Basic Indexing. Done!
one_d_array = np.arange(10, 60, 10)
print(f"This is 1D NumPy array with the values >> {one_d_array}")
print(f"This is the first element of the array >> {one_d_array[0]}")
print(f"This is the last element of the array >> {one_d_array[-1]}")

three_by_three_array = np.arange(1, 10).reshape(3, 3)
print(f"This is 2D NumPy array with the values\n ˅\n{three_by_three_array}")

"""for accessing element in specific cell you can use the following line
    or just like normal lists three_by_three_array[0][2] both will work
"""
print(
    f"This is the element in the second row and third column >> {three_by_three_array[0, 2]}"
)

"""We can also create the 2D array directly without reshaping just like the below example
    but I like the first method looks better for me"""

two_d_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"This is 2D NumPy array with the values\n ˅\n{two_d_array}")

"""2.Slicing 1D Arrays:
    Using the 1D array from the previous exercise ([10, 20, 30, 40, 50]), extract a subarray containing the elements from the second to the fourth position (inclusive).
    Extract the first three elements of the array.
    Extract the last two elements of the array.
    Extract every other element of the array.
    Reverse the array using slicing."""

# ToDo 2 Slicing 1D Arrays. Done!
sub_array = one_d_array[1:4]
print(
    f"Subarray from our 1D array containing the elements from the second to the fourth position (From 20 to 40) >> {sub_array}"
)

first_three = one_d_array[:3]
print(f"The first three elements of the 1D array >> {first_three}")

last_two = one_d_array[-2:]
print(f"The last two elements of the 1D array >> {last_two}")


other_elements = [i for i in one_d_array if i not in last_two]
print(f"The other element of the 1D array >> {other_elements}")

reversed_array = one_d_array[::-1]
print(f"Reversed array using slicing of the 1D array >> {reversed_array}")


"""3.Slicing 2D Arrays:
    Using the 2D array from the first exercise (the 3x3 array), extract the first row.
    Extract the second column.
    Extract the top-left 2x2 subarray.
    Extract the bottom-right 2x2 subarray."""

# ToDo 3 Slicing 2D Arrays. Done!
first_row = three_by_three_array[0]
print(f"2D array's first row >> {first_row}")

second_column = three_by_three_array[0:3, 1]
print(f"2D array's second column >> {second_column}")

top_left_sub_array = three_by_three_array[0:2, 0:2]
print(f"2D array's top-left 2x2 subarray \n ˅\n {top_left_sub_array}")

bottom_right_sub_array = three_by_three_array[1:3, 1:3]
print(f"2D array's bottom-right 2x2 subarray \n ˅\n {bottom_right_sub_array}")


"""4.Boolean Indexing:

    Create a 1D NumPy array with 10 random integers between 0 and 100.
    Create a boolean array that is True for elements greater than 50 and False otherwise.
    Use the boolean array to select only the elements from the original array that are greater than 50."""

# ToDo 4 Boolean Indexing. Done!
random_array = np.random.randint(1, 101, size=(1, 10))
print(f"1D NumPy array with 10 random integers between 0 and 100 >> {random_array}")
# The function nditer() is a helping function that can be used from very basic to very advanced iterations. It solves some basic issues which we face in iteration
boolean_array = [n > 50 for n in np.nditer(random_array)]
print(
    f"Boolean array that is True for elements greater than 50 and False otherwise >> {boolean_array}"
)

filtered_elements = random_array[0][boolean_array]
print(
    f"The elements from the original array that are greater than 50 >> {filtered_elements}"
)


"""5.Integer Array Indexing (Fancy Indexing):

    Create a 1D NumPy array with the values [100, 200, 300, 400, 500].
    Use a list of indices [0, 3, 1] to select and print the elements at these indices.
    Create a 2D NumPy array (e.g., 4x4). Use two lists of indices to select specific elements (e.g., select elements at rows [0, 2] and columns [1, 3])."""

# ToDo 5 Integer Array Indexing (Fancy Indexing). Done!
hundereds_array = np.arange(100, 501, 100)
print(f"1D NumPy array with the values >> {hundereds_array}")

indices_elements = [hundereds_array[n] for n in [0, 3, 1]]
print(f"The elements at [0, 3, 1] indices are >> {indices_elements}")

four_by_four_array = np.arange(1, 17).reshape(4, 4)
print(
    f"Elements at rows [0, 2] and columns [1, 3] \n ˅\n {four_by_four_array[np.r_[0, 2][:, None], np.c_[1, 3][:, None]]}"
)

row_indices = [0, 2]
col_indices = [1, 3]

selected_elements = four_by_four_array[np.ix_(row_indices, col_indices)]
print(selected_elements)

"""6.Reshaping Arrays:

    Create a 1D NumPy array with the numbers from 0 to 11.
    Reshape this array into a 3x4 2D array.
    Reshape the same 1D array into a 4x3 2D array.
    Try to reshape it into a 5x? array and see what happens (think about the -1 in reshape)."""

# ToDo 6 Reshaping Arrays. Done!
normal_array = np.arange(12)
print(f"1D NumPy array with the numbers from 0 to 11 >> {normal_array}")

reshaped_3x4 = normal_array.reshape(3, 4)
print(f"Reshaped array into a 3x4 2D array \n ˅\n {reshaped_3x4}")

reshaped_4x3 = normal_array.reshape(4, 3)
print(f"Reshaped array into a 4x3 2D array \n ˅\n {reshaped_4x3}")

reshaped_5x = normal_array.reshape((2, -1))
print(
    f"Reshaped array into a 5x? isn't possible cuz 12/5 == float == (ERROR) so we will use 2x-1 2D array instead  \n ˅\n {reshaped_5x}"
)


"""7.Flattening Arrays:

    Using one of the 2D arrays you created,
    flatten it into a 1D array using both the .flatten() method and the .ravel() method. 
    Observe if there's any difference in their output for this case."""

# ToDo 7 Flattening Arrays. Done!
flatten = three_by_three_array.flatten()
print(f"Flatten 2D into a 1D array using .flatten() method >> {flatten}")


ravel = three_by_three_array.ravel()
print(f"Raveled 2D into a 1D array using .ravel() method >> {ravel}")

"""You are wondering what's the difference between them right? both gives same output.
   I will show you the difference watch 😉"""

print(f"Our normal 3x3 array looks like that \n ˅\n {three_by_three_array}")

flatten[0] = 100

print(f"Flatten after updating first element >> {flatten}")

print(
    f"Our normal 3x3 array looks like that after updating flatten \n ˅\n {three_by_three_array}"
)

"""Our array didn't change because flatten is just a copy now let's see ravel"""

ravel[0] = 500

print(f"Ravel after updating first element >> {ravel}")

print(
    f"Our normal 3x3 array looks like that after updating ravel \n ˅\n {three_by_three_array}"
)

"""See now our array's first element changed because ravel isn't just a copy
   it's a view anything you will edit here will change the original value so be careful with it"""
