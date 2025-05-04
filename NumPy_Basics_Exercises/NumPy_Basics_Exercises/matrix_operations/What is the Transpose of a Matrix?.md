Imagine you have a matrix (which is just a rectangular array of numbers, like the 2D NumPy arrays we've been working with). The **transpose** of this matrix is basically a new matrix where the rows and columns of the original matrix are swapped.

Think of it like flipping the matrix over its main diagonal (the diagonal from the top-left corner to the bottom-right corner).

**Example:**

Let's say you have a matrix `A`:

```
A = [[1, 2, 3],
     [4, 5, 6]]
```

This is a 2x3 matrix (2 rows and 3 columns).

The transpose of matrix `A`, often denoted as `A`<sup>T</sup> or `A.T`, would be:

```
A.T = [[1, 4],
       [2, 5],
       [3, 6]]
```

Notice how the first row of `A` ([1, 2, 3]) became the first column of `A.T`, and the second row of `A` ([4, 5, 6]) became the second column of `A.T`. Similarly, the first column of `A` ([1, 4]) became the first row of `A.T`, and so on.

**How to do it in NumPy:**

NumPy makes finding the transpose of an array super easy! You can do it in a couple of ways:

1. **Using the `.T` attribute:** You can simply access the `.T` attribute of your NumPy array.
    
2. **Using the `np.transpose()` function:** You can also use the `np.transpose()` function, passing your array as an argument.
    

**Let's see it in action with Python code:**

```Python
import numpy as np

# Original matrix A
A = np.array([[1, 2, 3],
              [4, 5, 6]])

print("Original Matrix A:")
print(A)
print("Shape of A:", A.shape)  # (2, 3)

# Transpose using .T attribute
AT = A.T
print("\nTranspose of A (using .T):")
print(AT)
print("Shape of AT:", AT.shape) # (3, 2)

# Transpose using np.transpose() function
AT_alt = np.transpose(A)
print("\nTranspose of A (using np.transpose()):")
print(AT_alt)
print("Shape of AT_alt:", AT_alt.shape) # (3, 2)
```

### So np.T is a shortcut for np.transpose()? if yes, then in which cases would I feel I'm forced to use the function instead of the shortcut?

For most common matrix transpose operations, using `.T` is perfectly sufficient and often more convenient because it's shorter and more visually direct. 😊

However, there are situations where you would definitely want to use the full `np.transpose()` function:

1. **Working with arrays that have more than two dimensions:** The `.T` attribute **always** swaps the last two axes of an array. If you have a 3D array (or higher dimensions) and you want to swap axes other than the last two, you need to use `np.transpose()` and explicitly specify the order of the axes you want.
    
    Let's imagine you have a 3D array representing, say, a series of color images where the dimensions are (number of images, height, width, color channels). If you wanted to rearrange the dimensions, for example, to have (height, width, number of images, color channels), you would need to use `np.transpose()` and tell it the new order of the axes.
    
    Here's a simple example with a 3D array:
	```python
	import numpy as np
	
	# A 3D array with shape (2, 3, 4)
	arr_3d = np.arange(24).reshape((2, 3, 4))
	print("Original 3D array (shape: {}):\n{}".format(arr_3d.shape, arr_3d))
	
	# Using .T will swap the last two axes (3 and 4) -> (2, 4, 3)
	transposed_T = arr_3d.T
	print("\nTransposed with .T (shape: {}):\n{}".format(transposed_T.shape, transposed_T))
	
	# Using np.transpose() to swap the first and second axes -> (3, 2, 4)
	transposed_func = np.transpose(arr_3d, (1, 0, 2))
	print("\nTransposed with np.transpose((1, 0, 2)) (shape: {}):\n{}".format(transposed_func.shape, transposed_func))
	```
	In this example, `.T` gives you a different result than `np.transpose()` with specific axis order.
	
2. **When you want to be very explicit about the axes you are swapping, even for a 2D array:** While `.T` is clear for 2D arrays, using `np.transpose(A, (1, 0))` explicitly states that you're swapping axis 0 (rows) with axis 1 (columns). This can sometimes improve code readability, especially if the code is part of a larger, more complex operation.
    

So, in summary:

- For simple 2D matrix transposes, `.T` is a handy and common shortcut.
- For multi-dimensional arrays or when you need to specify a particular order of axes for transposition, you'll need to use the more versatile `np.transpose()` function.

### Okay, I understood the concept. But I'm not sure how the numbers that are passed to the function are used. How (1, 0, 2) reshaped to (3, 2, 4) are we adding the passed numbers to the original shape? But I guess no doesn't make sense, this would result (3, 3, 6). I thought of multiple methods, but couldn't find the correct answer

You're correct, the numbers in the tuple `(1, 0, 2)` passed to `np.transpose()` are **not** added to the original shape. Instead, they represent the **new order of the original axes**.

Let's break it down using our 3D array example:

```python
import numpy as np

# A 3D array with shape (2, 3, 4)
arr_3d = np.arange(24).reshape((2, 3, 4))
# Think of the axes as:
# Axis 0: size 2 (the "outermost" dimension, like different images)
# Axis 1: size 3 (the "middle" dimension, like height)
# Axis 2: size 4 (the "innermost" dimension, like width)

# np.transpose(arr_3d, (1, 0, 2))
```

The tuple `(1, 0, 2)` tells NumPy:

- **"For my new first axis (axis 0), I want to use the original axis at index 1."** The original axis at index 1 had a size of 3. So, the new first axis will have a size of 3.
- **"For my new second axis (axis 1), I want to use the original axis at index 0."** The original axis at index 0 had a size of 2. So, the new second axis will have a size of 2.
- **"For my new third axis (axis 2), I want to use the original axis at index 2."** The original axis at index 2 had a size of 4. So, the new third axis will have a size of 4.

Therefore, the new shape becomes `(3, 2, 4)`.

**Think of it like relabeling the dimensions:**

Imagine the original dimensions are labeled as Axis 0, Axis 1, and Axis 2. When you use `np.transpose(arr_3d, (1, 0, 2))`, you're saying:

- The dimension that was originally Axis 1 is now the first dimension.
- The dimension that was originally Axis 0 is now the second dimension.
- The dimension that was originally Axis 2 remains the third dimension.

**Example with a 2D array:**

Let's go back to a 2D array `A` with shape `(2, 3)`:
```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
# Axis 0: size 2 (rows)
# Axis 1: size 3 (columns)
```

When you do `A.T` or `np.transpose(A)`, it's equivalent to `np.transpose(A, (1, 0))`. This means:

- The new axis 0 will be the original axis 1 (size 3 - the original columns become the new rows).
- The new axis 1 will be the original axis 0 (size 2 - the original rows become the new columns).

So the new shape is `(3, 2)`.