import pandas as pd
import numpy as np

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
    "salary": [
        60000,
        75000,
        65000,
        80000,
        70000,
        78000,
        62000,
        72000,
        79000,
        np.nan,
    ],  # Added a NaN salary
    "years_experience": [2, 5, 3, 7, 1, 4, 2, 3, 6, 1],
}  # Added years of experience

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)


"""Basic Statistical Analysis Exercises:"""

"""1.Overall Descriptive Statistics:

Problem: Get the descriptive statistics (.describe()) for the numerical columns of the entire DataFrame.
Print the result."""

descriptive = df.describe()
print("Descriptive statistics for the numerical columns of the entire DataFrame")
print(descriptive)


"""2. Specific Overall Statistics:

    Problem: Calculate and print the following for the entire DataFrame:
        The median of the 'salary' column.
        The standard deviation of the 'years_experience' column.
        The total number of non-NaN values in the 'salary' column.
    Print each result clearly."""

salary_median = df["salary"].median()
print("The median of the 'salary' column")
print(salary_median)

experience_std = df["years_experience"].std()
print("The standard deviation of the 'years_experience' column")
print(experience_std)

total_non_NaN = df["salary"].count()
print("The total number of non-NaN values in the 'salary' column")
print(total_non_NaN)


"""3.Grouped Statistics (Mean and Count):

    Problem: Group the DataFrame by 'department'. 
    For each department, 
    calculate the mean 'salary' and the number of employees (count of non-NaN 'salary' values).
    Print the result."""

grouped_df = df.groupby("department").agg({"salary": ["mean", "count"]})
print(
    "The mean 'salary' and the number of employees (count of non-NaN 'salary' values)"
)
print(grouped_df)


"""4.Grouped Statistics (Multiple Aggregations):

    Problem: Group the DataFrame by 'department'. 
    For each department, calculate the minimum 'salary', 
    the maximum 'salary', and the average 'years_experience'.
    Print the result."""

multi_agg = df.groupby("department").agg(
    {"salary": ["min", "max"], "years_experience": "mean"}
)
print("The minimum 'salary', the maximum 'salary', and the average 'years_experience'")
print(multi_agg)
