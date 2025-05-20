import torch
import torch.nn as nn  # nn contains all the modules we'll use to build our network
import torch.nn.functional as F  # F contains activation functions and other useful operations


# Define the Neural Network class
class SimpleMLP(nn.Module):
    # The __init__ method defines the layers of our network
    def __init__(self):
        super(
            SimpleMLP, self
        ).__init__()  # Call the constructor of the parent class (nn.Module)

        # Input layer (784 units) to Hidden layer (128 units)
        # nn.Linear is a module that applies a linear transformation: y = xA^T + b
        # In our case, it's a fully connected layer where every input unit is connected to every output unit.
        self.fc1 = nn.Linear(
            28 * 28, 128
        )  # 28*28 = 784, because each MNIST image is 28x28 pixels.

        # Hidden layer (128 units) to Output layer (10 units)
        # The output layer has 10 units because there are 10 possible digits (0-9).
        self.fc2 = nn.Linear(128, 10)

    # The forward method defines how data flows through the network
    def forward(self, x):
        # Flatten the input image from 28x28 to a 1D vector of 784 elements
        # -1 means infer the batch size automatically.
        x = x.view(-1, 28 * 28)

        # Apply the first linear transformation (fc1)
        # Then apply the Rectified Linear Unit (ReLU) activation function
        # F.relu(x) is a common activation function that introduces non-linearity.
        # Without non-linearity, a deep network would just be a single linear transformation.
        x = F.relu(self.fc1(x))

        # Apply the second linear transformation (fc2)
        # No activation function is applied here yet, as it's common to apply
        # the activation (e.g., softmax for probabilities) later during the loss calculation.
        x = self.fc2(x)

        return x


# Create an instance of our network
model = SimpleMLP()

# Print the model architecture (useful for checking layers)
print(model)

# Move the model to the appropriate device (CPU or GPU)
# If a GPU is available, we'll use it for faster computation.
# Otherwise, we default to the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(
    device
)  # .to() moves the model's parameters and buffers to the specified device

print(f"\nModel moved to: {device}")
