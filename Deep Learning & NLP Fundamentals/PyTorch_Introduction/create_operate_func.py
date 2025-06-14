import torch

"""1. Tensor Creation

a) Create a 2D tensor of shape (3, 4) filled with random integers between 0 and 9 (inclusive).  
b) Convert this tensor to a NumPy array and print its shape.
"""
ts = torch.randint(0, 10, (3, 4))
print(ts)
print(ts.numpy().shape)


""" 2. Tensor Operations

a) Create another 2D tensor of the same shape, filled with ones (as floats).  
b) Perform element-wise multiplication between the two tensors.  
c) Compute the sum of all elements in the result.
"""
ts1 = torch.ones((3, 4))
ts2 = ts1 * ts
print(ts2)
print(ts2.sum().item())


"""3. Slicing and Indexing

a) Extract the second row from your original random tensor.  
b) Extract the last column from your original random tensor.
"""
ts = torch.randint(0, 10, (3, 4))
print(ts)
print(ts[1, :])
print(ts[:, 3])


"""4. Reshaping

a) Reshape your original random tensor into a 1D tensor (flatten it).  
b) Reshape it back to shape (4, 3).
"""
ts = torch.randint(0, 10, (3, 4))
print(ts)
print(ts.view(-1))
print(ts.view(4, 3))


"""5. Automatic Differentiation

a) Create a tensor `x` with value 5.0 and set `requires_grad=True`.  
b) Define a function: `y = 3 * x ** 2 + 2 * x + 1`  
c) Compute the loss as: `loss = (y - 50) ** 2`  
d) Perform a backward pass to compute the gradient of the loss with respect to `x`.  
e) Print the value of `x.grad`.
"""
ts = torch.tensor(5.0, requires_grad=True)
y = 3 * ts**2 + 2 * ts + 1
loss = (y - 50) ** 2
loss.backward()
print(ts.grad.item())


"""6. Bonus: torch.no_grad()

a) Use a `with torch.no_grad():` block to compute a new value of `y` for `x = 10.0` (no gradient tracking).  
b) Try to call `.backward()` on this new value and observe what happens.
"""
ts = torch.tensor(10.0, requires_grad=True)
with torch.no_grad():
    y = 3 * ts**2 + 2 * ts + 1
print(y)
# This will raise an error because y is not a leaf node
print(y.backward())
