from sklearn.model_selection import train_test_split
import numpy as np  # Using numpy for a simple example

# Sample data (Imagine X is features, y is the target)
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])
y = np.array([0, 1, 0, 1, 0, 1, 0, 1])  # Binary target for simplicity

print("Original X:\n", X)
print("Original y:\n", y)

# Split the data
# test_size=0.25 means 25% of the data goes to the test set
# random_state ensures the split is the same each time you run it (reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=99
)

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("\nX_train:\n", X_train)
print("X_test:\n", X_test)  # Notice these were not in X_train
