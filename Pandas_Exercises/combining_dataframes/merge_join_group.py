import pandas as pd

data = {
    "employee_id": [101, 102, 103, 104, 105, 106, 107, 108],
    "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Heidi"],
    "department": [
        "Sales",
        "IT",
        "Sales",
        "IT",
        "Marketing",
        "IT",
        "Sales",
        "Marketing",
    ],
    "salary": [60000, 75000, 65000, 80000, 70000, 78000, 62000, 72000],
    "hire_year": [2020, 2018, 2019, 2017, 2021, 2019, 2020, 2021],
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# # Group by 'department' and calculate the mean salary for each department
# department_salary_mean = df.groupby("department")["salary"].mean()
# print("\nMean salary per department:")
# print(department_salary_mean)

# # Group by 'department' and calculate the total number of employees in each department
# department_employee_count = df.groupby("department")[
#     "employee_id"
# ].count()  # Or df.groupby('department').size()
# print("\nNumber of employees per department:")
# print(department_employee_count)

# # Group by 'department' and 'hire_year' and calculate the mean salary
# department_year_salary_mean = df.groupby(["department", "hire_year"])["salary"].mean()
# print("\nMean salary per department and hire year:")
# print(department_year_salary_mean)

# # Group by 'department' and find the min, max, and mean salary for each department
# department_salary_agg = df.groupby("department")["salary"].agg(["min", "max", "mean"])
# print("\nMin, Max, and Mean salary per department:")
# print(department_salary_agg)

# You can also apply different aggregations to different columns using a dictionary in agg()
department_agg_different = df.groupby("department").agg(
    {"salary": "mean", "hire_year": "min"}
)
print("\nMean salary and Min hire year per department:")
print(department_agg_different)
