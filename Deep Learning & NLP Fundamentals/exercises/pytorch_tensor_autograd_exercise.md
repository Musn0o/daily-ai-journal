# PyTorch Tensors & Autograd Practice Exercise

**Instructions:**  
Write your code in a new Python file or Jupyter notebook. Use only PyTorch and NumPy as needed.  
**Do not look up the solution until you’ve tried your best!**

---

## 1. Tensor Creation

a) Create a 2D tensor of shape (3, 4) filled with random integers between 0 and 9 (inclusive).  
b) Convert this tensor to a NumPy array and print its shape.

---

## 2. Tensor Operations

a) Create another 2D tensor of the same shape, filled with ones (as floats).  
b) Perform element-wise multiplication between the two tensors.  
c) Compute the sum of all elements in the result.

---

## 3. Slicing and Indexing

a) Extract the second row from your original random tensor.  
b) Extract the last column from your original random tensor.

---

## 4. Reshaping

a) Reshape your original random tensor into a 1D tensor (flatten it).  
b) Reshape it back to shape (4, 3).

---

## 5. Automatic Differentiation

a) Create a tensor `x` with value 5.0 and set `requires_grad=True`.  
b) Define a function: `y = 3 * x ** 2 + 2 * x + 1`  
c) Compute the loss as: `loss = (y - 50) ** 2`  
d) Perform a backward pass to compute the gradient of the loss with respect to `x`.  
e) Print the value of `x.grad`.

---

## 6. Bonus: torch.no_grad()

a) Use a `with torch.no_grad():` block to compute a new value of `y` for `x = 10.0` (no gradient tracking).  
b) Try to call `.backward()` on this new value and observe what happens.

---

**Good luck!**  
Try to reason through each step before running the code.