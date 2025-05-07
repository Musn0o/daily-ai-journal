import pandas as pd
import io


employees_csv = """
employee_id,name,department_id,hire_date
101,Alice,10,2020-01-15
102,Bob,20,2018-07-20
103,Charlie,10,2019-05-10
104,David,20,2017-11-01
105,Eve,30,2021-03-25
106,Frank,20,2019-09-15
107,Grace,10,2020-08-01
108,Heidi,30,2021-06-10"""


salaries_csv = """
employee_id,salary,performance_score
101,60000,85
102,75000,90
103,65000,88
104,80000,92
105,70000,87
106,78000,91
107,62000,86
109,70000,89"""  # Added employee_id 109 who is not in employees_csv


"""Comprehensive Pandas Exercises:"""

"""1. Merge, Filter, and Select:

Load employees_csv and salaries_csv into two DataFrames.
Problem: Merge these two DataFrames based on the employee_id. 
Keep only the employees who are in employees_csv (Hint: think about merge types!). 
From the merged data, filter for employees who were hired in 2020 or later AND have a performance score above 87. 
Finally, select only the 'name', 'department_id', and 'salary' columns for these filtered employees.
Print the resulting DataFrame."""

employees_df = pd.read_csv(io.StringIO(employees_csv))
print("Employees DataFrame")
print(employees_df)

salaries_df = pd.read_csv(io.StringIO(salaries_csv))
print("Salaries DataFrame")
print(salaries_df)

merged_dfs = pd.merge(employees_df, salaries_df, on="employee_id", how="left")
print(
    "Merged DataFrames based on the employee_id. Only the employees who are in employees_csv exist despite that Heidi got NaN in salaries"
)
print(merged_dfs)

merged_filtered = merged_dfs.loc[
    (merged_dfs["hire_date"] >= "2020-01-01") & (merged_dfs["performance_score"] >= 87)
]
print(
    "None from the hired in 2020 got score more than 87 if it's >= Eve would pop up right"
)
print(merged_filtered)

specific_cols = merged_filtered[["name", "department_id", "salary"]]
print("The 'name', 'department_id', and 'salary' columns for these filtered employees")
print(specific_cols)


"""2.Merge, Group, and Sort:

    Load employees_csv and salaries_csv into two DataFrames.
    Problem: Merge these two DataFrames using an inner merge on employee_id. 
    Group the merged data by department_id. For each department, 
    calculate the average salary and the total number of employees. 
    Sort the results by the average salary in descending order.
    Print the resulting DataFrame."""

employees = pd.read_csv(io.StringIO(employees_csv))
print("Employees DataFrame")
print(employees)

salaries = pd.read_csv(io.StringIO(salaries_csv))
print("Salaries DataFrame")
print(salaries)

merged = pd.merge(employees, salaries, how="inner", on="employee_id")
print("Merged the DataFrames using an inner merge on employee_id")
print(merged)

grouped = merged.groupby("department_id")["salary"].agg(["mean", "count"])
print("Average salary and the total number of employees for each department")
print(grouped)

sorted = grouped.sort_values(by="mean", ascending=False)
print("Sorted results by the average salary in descending order")
print(sorted)


"""3.Load, Filter, Group, and Analyze:

    Load employees_csv and salaries_csv into two DataFrames.
    merge them first (using employee_id)
    Problem: Filter the DataFrame to include only employees hired before 2020. 
    Group this filtered data by department_id. 
    For each of these older employee groups, find the minimum and maximum hire date and the average salary.
    Print the resulting DataFrame."""

df_emp = pd.read_csv(io.StringIO(employees_csv))
print(df_emp)

df_sal = pd.read_csv(io.StringIO(salaries_csv))
print(df_sal)

m_emp_sal = pd.merge(df_emp, df_sal, on="employee_id")
old_employees = m_emp_sal[m_emp_sal["hire_date"] < "2020-01-01"]
print("employees hired before 2020")
print(old_employees)

dep_grouped = old_employees.groupby("department_id").agg(
    {"hire_date": ["min", "max"], "salary": "mean"}
)
print("Older employee minimum and maximum hire date and the average salary")
print(dep_grouped)
