import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler  # You'll need this!
from sklearn.metrics import mean_squared_error, r2_score  # New for evaluation!


"""Harder Linear Regression Exercise: Employee Salary Prediction with Evaluation

We'll use our df_cleaned DataFrame. Your task is to build a Linear Regression model to predict salary based on years_experience and performance_score.

Here's the setup for our data (using df_cleaned from our previous exercises):"""

# Re-create our cleaned DataFrame for this exercise
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
    "education_level": [
        "Bachelor",
        "Master",
        "PhD",
        "Master",
        "Bachelor",
        "PhD",
        "Bachelor",
        "Master",
        "PhD",
        "Bachelor",
    ],
}

df = pd.DataFrame(data)
df_cleaned = df.dropna(subset=["salary"]).copy()  # Drop the row with NaN salary

# Define X (features) and y (target) for this exercise
X = df_cleaned[["years_experience", "performance_score"]]  # Multiple features now!
y = df_cleaned["salary"]

print("Features (X head):\n", X.head())
print("\nTarget (y head):\n", y.head())
print("\n----------------------------------------------------\n")

"""Your Harder Task: Build, Train, Predict, and Evaluate!"""

"""1.Train-Test Split:
    Split the X and y data into training and testing sets. 
    Use test_size=0.2 and random_state=42."""

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

"""2.Feature Scaling:

    Instantiate a StandardScaler.
    Apply the StandardScaler to your training features (X_train) using fit_transform().
    Apply the same fitted StandardScaler to your test features (X_test) using transform().
    (Remember, no fit() on test data!)"""

# Instantiate a StandardScaler
scaler = StandardScaler()
# Apply the StandardScaler to the training features
X_train_scaled = scaler.fit_transform(X_train)
# Apply the same fitted StandardScaler to the test features
X_test_scaled = scaler.transform(X_test)

"""quickly explain why we are using fit_transform on X_train and only transform on X_test?"""
"""
The reason we use `fit_transform` on `X_train` and `transform` on `X_test` is to ensure that the scaling process is consistent across both the training and testing sets.
This is important because the scaling process can affect the performance of the model, and we want to avoid any differences between the two sets that could lead to biased results. 
By using `fit_transform` on `X_train`, we ensure that the mean and standard deviation of the features are calculated based on the training data, and then we apply this scaling to the training data.
On the other hand, we use `transform` on `X_test` because we don't want to scale the test data, as we want to use it to make predictions on new data that hasn't been scaled.
By using `fit_transform` on `X_train` and `transform` on `X_test`, we ensure that the scaling process is consistent across both the training and testing sets,
which is important because the scaling process can affect the performance of the model, and we want to avoid any differences between the two sets that could lead to biased results.
the model is trained on scaled data and evaluated on scaled data. This helps to avoid any differences between the two sets that could lead to biased results.
"""

"""3.Model Training:

    Instantiate a LinearRegression model.
    Train the model using your scaled training features (X_train_scaled) and your training target (y_train)."""

# Instantiate a LinearRegression model
lr_model = LinearRegression()
# Train the model using the scaled training features and the training target
lr_model.fit(X_train_scaled, y_train)

"""4.Make Predictions:

    Use the trained model to make predictions on your scaled test features (X_test_scaled). Store these as y_pred."""

# Use the trained model to make predictions on the scaled test features
y_pred = lr_model.predict(X_test_scaled)

"""5.Model Evaluation:

    Calculate the Mean Squared Error (MSE) between your actual y_test and your y_pred.
    Calculate the R-squared (R2) score between your actual y_test and your y_pred.
    Print both scores, along with the model's coefficients and intercept (as we did in the example)."""

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
# Calculate the R-squared (R2) score
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R2): {r2}")
# Print the model's coefficients and intercept
print(f"Coefficients: {lr_model.coef_}")
print(f"Intercept: {lr_model.intercept_}")

"""What coeficients and intercept do are good for?
The coefficients and intercept of a linear regression model are important for understanding the relationship between the features and the target variable.
for example : if the coefficient of 'years_experience' is positive, it means that as the years of experience increases, the salary also increases.
if the coefficient of 'performance_score' is negative, it means that as the performance score decreases, the salary also decreases.
the intercept is the value of the target variable when all the features are zero.
it is the baseline value of the target variable.
for example : if the intercept is 10000, it means that when all the features are zero, the salary is 10000.

"""
