import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression  # We'll use this simple model

# Our Employee DataFrame (from previous exercises)
data = {
    "employee_id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "department": [
        "Sales",
        "IT",
        "Sales",
        "IT",
        "Marketing",
        "IT",
        "Sales",
        "Marketing",
        "IT",
        "Sales",
    ],
    "salary": [60000, 75000, 65000, 80000, 70000, 78000, 62000, 72000, 79000, np.nan],
    "years_experience": [2, 5, 3, 7, 1, 4, 2, 3, 6, 1],
    "performance_score": [85, 90, 88, 92, 87, 91, 86, 89, 93, 80],
}

df = pd.DataFrame(data)

# Handle the missing salary value for now (we'll cover imputation properly later)
# For simplicity, let's drop the row with NaN for this exercise
df_cleaned = df.dropna(subset=["salary"]).copy()

# Define X (features) and y (target)
# Features (X) will be 'years_experience' and 'performance_score'
X = df_cleaned[["years_experience", "performance_score"]]
# Target (y) will be 'salary'
y = df_cleaned["salary"]

print("Features (X) head:\n", X.head())
print("\nTarget (y) head:\n", y.head())


"""Exercise 1: Train-Test Split

    Problem: Split the X (features) and y (target) data into training and testing sets.
        Allocate 30% of the data to the test set (test_size=0.3).
        Use random_state=42 for reproducibility.
    Task:
        Perform the train-test split.
        Print the shapes of X_train, X_test, y_train, and y_test to verify the split."""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(
    "\nShapes of X_train, X_test, y_train, and y_test:\n",
    X_train.shape,
    X_test.shape,
    y_train.shape,
    y_test.shape,
)

"""Exercise 2: K-Fold Cross-Validation

    Problem: Perform 5-fold cross-validation on our data using a LinearRegression model.
    Task:
        Instantiate a LinearRegression model.
        Use cross_val_score to get 5 scores for the model's performance on the X and y data.
        For regression, a common scoring metric is 'r2' (R-squared, which measures how well future samples are likely to be predicted by the model).
        Print the individual cross-validation scores.
        Calculate and print the mean of these scores to get an overall estimate of the model's performance.
        Note: You can use the 'r2' scoring metric by passing it as an argument to cross_val_score.
        """
model = LinearRegression()
scores = cross_val_score(model, X, y, scoring="r2", cv=5)
# or to avoid the nan errors uncomment the following line
# scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5)
print("\nIndividual cross-validation scores:\n", scores)
print("\nMean of cross-validation scores:\n", scores.mean())
