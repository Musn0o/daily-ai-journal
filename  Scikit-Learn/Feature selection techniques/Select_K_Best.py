import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
)  # f_classif is for classification tasks

# Set a random seed for reproducibility
np.random.seed(42)

# Create a synthetic dataset with various features
# 'age', 'income', 'daily_steps' are designed to be more relevant
# 'num_children' is moderately relevant
# 'zip_code' is designed to be irrelevant noise
X_exercise = pd.DataFrame(
    {
        "age": np.random.randint(20, 60, 100),
        "income": np.random.normal(50000, 15000, 100),
        "num_children": np.random.randint(0, 5, 100),
        "zip_code": np.random.randint(10000, 99999, 100),  # Irrelevant feature
        "daily_steps": np.random.normal(8000, 2000, 100),
    }
)

# Create a binary target 'y_exercise' (0 or 1 for 'buys_product')
# The target is primarily influenced by 'age', 'income', and 'daily_steps'
y_exercise = (
    (X_exercise["age"] * 0.1)
    + (X_exercise["income"] * 0.0001)
    + (X_exercise["daily_steps"] * 0.0005)
    + np.random.randn(100) * 0.5  # Add some random noise
)

# Convert to a binary target (0 or 1) by comparing to the median
y_exercise = (y_exercise > y_exercise.median()).astype(int)

print("Original Features (X_exercise head):\n", X_exercise.head())
print("\nOriginal Target (y_exercise head):\n", y_exercise.head())
print("\n----------------------------------------------------\n")

"""
Problem: You have a dataset with several features,
and you want to identify and select the top 3 most statistically relevant features to predict a binary outcome (like "buys_product").
One of the features (zip_code) is intentionally designed to be irrelevant. Dataset: We'll create a synthetic dataset for this task.

Task:

1.Create a SelectKBest instance to select the top 3 features. Use f_classif as the score_func because y_exercise is a classification target (0 or 1).
2.fit the SelectKBest instance to X_exercise and y_exercise.
3.Print the scores calculated by SelectKBest for all features. (Hint: selector.scores_)
4.Print the names of the selected features. (Hint: selector.get_support() returns a boolean mask which you can use with X_exercise.columns).
5.Transform X_exercise using the fitted selector to get X_selected (the DataFrame with only the chosen features).
6.Print the shape of the original X_exercise and the shape of X_selected to observe the dimensionality reduction.
"""
# Solution
selector = SelectKBest(score_func=f_classif, k=3)
selector.fit(X_exercise, y_exercise)
print("Scores calculated by SelectKBest:\n", selector.scores_)
print("Selected Features:\n", X_exercise.columns[selector.get_support()])
X_selected = selector.transform(X_exercise)
print("Shape of Original X_exercise:", X_exercise.shape)
print("Shape of X_selected:", X_selected.shape)
