import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations to apply to the images
# We'll convert images to PyTorch tensors and normalize them.
# transforms.ToTensor() converts a PIL Image or numpy.ndarray to a tensor.
# transforms.Normalize() normalizes a tensor image with mean and standard deviation.
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Mean and Std Dev for MNIST
    ]
)

# Load the training dataset
# root='data' specifies where to download the dataset
# train=True means we want the training split
# download=True will download the dataset if it's not already there
# transform=transform applies our defined transformations
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

# Load the test dataset
# train=False means we want the test split
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# Create DataLoaders for easy batching and shuffling
# DataLoader wraps a dataset and provides an iterable over it.
# batch_size defines how many samples per batch to load.
# shuffle=True shuffles the data at every epoch (good for training).
# num_workers specifies how many subprocesses to use for data loading (0 means main process).
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")
print(
    f"Shape of a single image tensor: {train_dataset[0][0].shape}"
)  # [0] for image, [0] for the first image, [0] for the tensor itself
