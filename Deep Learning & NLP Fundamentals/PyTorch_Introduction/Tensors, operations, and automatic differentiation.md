### **1. Tensors: The Building Blocks of PyTorch**

In PyTorch, the fundamental data structure is the **Tensor**. You can think of a Tensor as a multi-dimensional array, very similar to NumPy arrays, but with two key advantages:

1. **GPU Acceleration:** Tensors can seamlessly run computations on Graphics Processing Units (GPUs), which are designed for parallel processing and are essential for the speed of deep learning.
2. **Automatic Differentiation (`autograd`):** PyTorch can automatically calculate gradients for operations performed on Tensors, which is the cornerstone of how neural networks learn (backpropagation).

Tensors can hold scalars (0-D tensor), vectors (1-D tensor), matrices (2-D tensor), or higher-dimensional data (e.g., a 3-D tensor for images, or 4-D for a batch of images).

**How to Create Tensors:**

```python
import torch
import numpy as np

# 1. Creating a scalar (0-D tensor)
scalar_tensor = torch.tensor(7)
print("Scalar Tensor:", scalar_tensor)
print("Scalar Tensor Shape:", scalar_tensor.shape) # Shape is empty tuple for scalar

# 2. Creating a vector (1-D tensor)
vector_tensor = torch.tensor([1, 2, 3])
print("\nVector Tensor:", vector_tensor)
print("Vector Tensor Shape:", vector_tensor.shape)

# 3. Creating a matrix (2-D tensor)
matrix_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("\nMatrix Tensor:\n", matrix_tensor)
print("Matrix Tensor Shape:", matrix_tensor.shape)

# 4. Creating a 3-D tensor
three_d_tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("\n3-D Tensor:\n", three_d_tensor)
print("3-D Tensor Shape:", three_d_tensor.shape)

# 5. Creating tensors from NumPy arrays
numpy_array = np.array([10, 20, 30])
tensor_from_numpy = torch.from_numpy(numpy_array)
print("\nTensor from NumPy:", tensor_from_numpy)

# 6. Creating tensors with specific properties (zeros, ones, random)
zeros_tensor = torch.zeros(2, 3) # 2 rows, 3 columns of zeros
print("\nZeros Tensor (2x3):\n", zeros_tensor)

ones_tensor = torch.ones(3, 2) # 3 rows, 2 columns of ones
print("\nOnes Tensor (3x2):\n", ones_tensor)

random_tensor = torch.rand(4, 4) # 4x4 tensor with random values between 0 and 1
print("\nRandom Tensor (4x4):\n", random_tensor)

# You can also specify the data type (dtype)
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
print("\nInt Tensor:", int_tensor)
```


---

### **2. Tensor Operations**

Once you have Tensors, you can perform a wide range of operations on them, very similar to how you'd work with NumPy arrays.

```python
import torch

# Let's create a couple of tensors for operations
tensor_a = torch.tensor([[1, 2], [3, 4]])
tensor_b = torch.tensor([[5, 6], [7, 8]])

print("Tensor A:\n", tensor_a)
print("Tensor B:\n", tensor_b)

# 1. Element-wise Addition
add_result = tensor_a + tensor_b
print("\nElement-wise Addition (A + B):\n", add_result)
# Or using torch.add()
add_result_func = torch.add(tensor_a, tensor_b)
print("Element-wise Addition (torch.add(A, B)):\n", add_result_func)

# 2. Element-wise Subtraction
sub_result = tensor_a - tensor_b
print("\nElement-wise Subtraction (A - B):\n", sub_result)

# 3. Element-wise Multiplication (Hadamard product)
mul_result_element_wise = tensor_a * tensor_b
print("\nElement-wise Multiplication (A * B):\n", mul_result_element_wise)

# 4. Matrix Multiplication (Dot product)
# Requirements: Inner dimensions must match (e.g., (m, n) @ (n, p) -> (m, p))
matrix_mul_result = torch.matmul(tensor_a, tensor_b)
print("\nMatrix Multiplication (A @ B):\n", matrix_mul_result)
# Or using the '@' operator (Python 3.5+ for matrix multiplication)
matrix_mul_operator = tensor_a @ tensor_b
print("Matrix Multiplication (A @ B with @ operator):\n", matrix_mul_operator)

# 5. Reshaping Tensors
# view() returns a new tensor with the same data as the self tensor but with a different shape.
# If you modify the new tensor, the original tensor will also be modified.
tensor_to_reshape = torch.rand(2, 3)
print("\nOriginal Tensor (2x3):\n", tensor_to_reshape)
reshaped_tensor_view = tensor_to_reshape.view(3, 2) # Reshape to 3 rows, 2 columns
print("Reshaped Tensor (view):\n", reshaped_tensor_view)
# -1 in view() or reshape() means PyTorch infers the size
reshaped_tensor_infer = tensor_to_reshape.view(-1) # Flatten the tensor
print("Flattened Tensor (view with -1):\n", reshaped_tensor_infer)

# reshape() also changes the shape, but it can return a copy if the data is not contiguous
reshaped_tensor_reshape = tensor_to_reshape.reshape(1, 6)
print("Reshaped Tensor (reshape):\n", reshaped_tensor_reshape)

# 6. Slicing and Indexing (similar to NumPy)
sliced_tensor = matrix_tensor[0, 1] # Access element at row 0, column 1
print("\nSliced Element (matrix_tensor[0, 1]):", sliced_tensor)

row_slice = matrix_tensor[1, :] # Access all elements in row 1
print("Row Slice (matrix_tensor[1, :]):", row_slice)

column_slice = matrix_tensor[:, 0] # Access all elements in column 0
print("Column Slice (matrix_tensor[:, 0]):", column_slice)

# 7. Item access (for single-element tensors)
single_element_tensor = torch.tensor([42])
print("\nSingle Element Tensor:", single_element_tensor)
print("Accessed Item:", single_element_tensor.item()) # Converts to a standard Python number
```

### **3. Automatic Differentiation (`autograd`)**

This is where PyTorch truly shines and enables the "learning" in deep learning. `autograd` is PyTorch's powerful engine for automatically computing gradients of operations on Tensors.

**Why is it crucial?** Neural networks learn by adjusting their internal parameters (weights and biases) based on the "gradient" of a `loss function` with respect to those parameters. The gradient essentially tells us the direction and magnitude to change the parameters to minimize the loss. Calculating these gradients manually for complex networks would be incredibly tedious and error-prone. `autograd` does it for us!

**How it works:**

1. **`requires_grad=True`**: When you create a tensor, you can specify `requires_grad=True`. This tells PyTorch to track all operations performed on this tensor so that it can compute gradients later. These are typically your model's parameters (weights and biases).
2. **Computational Graph:** As operations are performed on tensors with `requires_grad=True`, PyTorch implicitly builds a dynamic computational graph in the background. This graph records how the output (e.g., the loss) was computed from the inputs.
3. **`loss.backward()`**: When you call `.backward()` on a scalar tensor (usually your loss function's output), PyTorch traverses this computational graph backward from the loss, using the chain rule to compute the gradients of the loss with respect to every tensor that had `requires_grad=True`.
4. **`.grad` attribute**: After `.backward()` is called, the gradients are accumulated in the `.grad` attribute of the respective tensors.

**Example: Simple Linear Function**

Let's imagine a very simple neural network: `y = w*x + b`, where `w` and `b` are our parameters (weights and biases) that we want to learn.

```python
import torch

# 1. Define our input and parameters (tensors with requires_grad=True)
# We need to compute gradients with respect to 'w' and 'b' to update them during training.
x = torch.tensor(2.0)
w = torch.tensor(3.0, requires_grad=True) # Parameters we want to optimize
b = torch.tensor(1.0, requires_grad=True) # Parameters we want to optimize

print("Initial w:", w)
print("Initial b:", b)

# 2. Perform a forward pass (build the computational graph)
y = w * x + b
print("\nForward pass: y =", y) # Should be 3 * 2 + 1 = 7

# Let's say our target output was 10.0, and we want to minimize the squared error.
# 3. Calculate the loss
target = torch.tensor(10.0)
loss = (y - target)**2
print("Loss:", loss) # (7 - 10)^2 = (-3)^2 = 9

# 4. Perform backward pass (compute gradients)
# This calculates d(loss)/dw and d(loss)/db
loss.backward()

# 5. Inspect gradients
# The gradient of loss with respect to w (d(loss)/dw)
# For loss = (wx + b - target)^2
# d(loss)/dw = 2 * (wx + b - target) * x
# d(loss)/dw = 2 * (7 - 10) * 2 = 2 * (-3) * 2 = -12
print("\nGradient of loss w.r.t. w (w.grad):", w.grad)

# The gradient of loss with respect to b (d(loss)/db)
# d(loss)/db = 2 * (wx + b - target) * 1
# d(loss)/db = 2 * (7 - 10) * 1 = 2 * (-3) * 1 = -6
print("Gradient of loss w.r.t. b (b.grad):", b.grad)

# Note: Gradients are accumulated. If you run loss.backward() again,
# the gradients will be added to the existing ones. So, you usually
# zero them out before each new backward pass in a training loop.
# print("\nRunning backward again (gradients accumulate):")
# loss.backward()
# print("w.grad after 2nd backward:", w.grad) # Will be -24

# 6. Using torch.no_grad()
# Sometimes you want to perform operations without tracking gradients (e.g., during model evaluation/inference).
# You can use torch.no_grad() context manager.
print("\nOperations within torch.no_grad():")
with torch.no_grad():
    new_y = w * x + b
    print("New y calculated without gradient tracking:", new_y)
    # If you try to call .backward() here, it won't work on new_y if its computation wasn't tracked.
    # Also, w.grad and b.grad would remain unchanged if you tried to compute through them.
```


This covers the essential concepts of Tensors, their operations, and the critical role of `autograd` in PyTorch. These are the fundamental building blocks for creating and training any neural network.

Take your time to run the code, experiment with different tensor creations and operations, and especially try to grasp how `autograd` works with the simple example.

Let me know if anything is unclear or if you have any questions!