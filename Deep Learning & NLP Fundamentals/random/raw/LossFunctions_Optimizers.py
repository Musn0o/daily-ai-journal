import torch
import torch.nn as nn
import torch.nn.functional as F  # F contains activation functions and other useful operations
import torch.optim as optim  # We'll import optimizers from here
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # For plotting the loss curve later

# --- (Previous code for dataset loading and model definition) ---

# Define transformations (copy-pasted for completeness, assuming you have it)
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# Define the SimpleMLP model (copy-pasted for completeness)
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Instantiate the model and move it to the device
model = SimpleMLP()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------------------------------------------------
# NEW CODE FOR LOSS FUNCTION AND OPTIMIZER

# Define the Loss Function (Criterion)
# nn.CrossEntropyLoss is suitable for multi-class classification.
criterion = nn.CrossEntropyLoss()

# Define the Optimizer
# It takes the model's parameters and a learning rate.
# model.parameters() tells the optimizer which values to adjust.
# lr (learning rate) is a hyperparameter that controls how big of a step the optimizer takes
# during each update. A smaller lr means slower but potentially more stable learning.
# A larger lr means faster but potentially unstable learning (overshooting the minimum).
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Loss Function: {criterion}")
print(f"Optimizer: {optimizer}")
print(f"Learning Rate: {optimizer.defaults['lr']}")
