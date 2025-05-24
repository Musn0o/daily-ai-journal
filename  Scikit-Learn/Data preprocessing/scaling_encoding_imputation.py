import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
)  # MinMaxScaler is not strictly needed for exercises but good to have
from sklearn.model_selection import train_test_split

# Our updated Employee DataFrame with NaN and a new categorical column 'education_level'
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
    "performance_score": [85, 90, 88, 92, np.nan, 91, 86, 89, 93, 80],  # Added NaN here
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
print("Original DataFrame:\n", df)

# Let's drop the row with NaN in salary for now, as it's our target.
# For simplicity in preprocessing exercises, we'll primarily focus on features.
df_cleaned = df.dropna(subset=["salary"]).copy()

"""Preprocessing Exercises:"""
"""Exercise 1: Imputation

    Problem: The performance_score column in df_cleaned has a missing value. Impute this missing value using the mean strategy.
    Task:
        Select the performance_score column (and years_experience to keep it 2D for SimpleImputer as it expects 2D array).
        Create a SimpleImputer instance with strategy='mean'.
        fit_transform the imputer on the selected column(s).
        Print the performance_score column from the original df_cleaned and then print the imputed performance_score to show the change."""

# Solution
imputer = SimpleImputer(strategy="mean")
df_cleaned["performance_score"] = imputer.fit_transform(
    df_cleaned[["performance_score", "years_experience"]]
)
print("Original Performance Score:\n", df["performance_score"])
print("Performance Score after Imputation:\n", df_cleaned["performance_score"])


"""Exercise 2: Encoding

    Problem: The department and education_level columns are categorical. Encode them appropriately.
    Task:
        One-Hot Encode the department column. Remember to set sparse_output=False for easier viewing and handle_unknown='ignore'.
        Print a portion of the resulting encoded array/DataFrame and its new column names.
        Ordinal Encode the education_level column.
        Make sure to define the correct order: ['Bachelor', 'Master', 'PhD'].
        Print a portion of the resulting encoded array/DataFrame."""

# Solution
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
df_cleaned_encoded = encoder.fit_transform(df_cleaned[["department"]])
print("Encoded Department:\n", df_cleaned_encoded)
print("Encoded Department Column Names:\n", encoder.get_feature_names_out())

encoder = OrdinalEncoder(categories=[["Bachelor", "Master", "PhD"]])
df_cleaned_encoded = encoder.fit_transform(df_cleaned[["education_level"]])
print("Encoded Education Level:\n", df_cleaned_encoded)
print("Encoded Education Level Column Names:\n", encoder.get_feature_names_out())

"""Exercise 3: Scaling (with Train-Test Split in mind)

    Problem: The years_experience column is numerical but on a different scale than other features might be.
    Apply StandardScaler to it. 
    More importantly, demonstrate the correct workflow of fitting on training data and transforming both training and testing data.
    Task:
        Select years_experience from df_cleaned as your feature X_scale_example.
        Create a dummy target y_scale_example (e.g., just df_cleaned['salary']) to enable train_test_split.
        Perform a train_test_split on X_scale_example and y_scale_example (e.g., test_size=0.3, random_state=42).
        Instantiate a StandardScaler.
        fit_transform the StandardScaler on X_train (the training set of years_experience).
        transform only X_test (the test set of years_experience) using the same fitted scaler.
        Print the first few values of X_train_scaled and X_test_scaled to observe the transformation."""

# Solution
X_scale_example = df_cleaned[["years_experience"]]
y_scale_example = df_cleaned["salary"]
X_train, X_test, y_train, y_test = train_test_split(
    X_scale_example, y_scale_example, test_size=0.3, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("First few values of X_train_scaled:\n", X_train_scaled[:5])
print("First few values of X_test_scaled:\n", X_test_scaled[:5])
