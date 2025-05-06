import pandas as pd
import io

csv_data = """
employee_id,name,department,salary,hire_date
101,Alice,Sales,60000,2020-01-15
102,Bob,IT,75000,2018-07-20
103,Charlie,Sales,65000,2019-05-10
104,David,IT,80000,2017-11-01
105,Eve,Marketing,70000,2021-03-25
106,Frank,IT,78000,2019-09-15
107,Grace,Sales,62000,2020-08-01
"""


"""Pandas Filtering, Sorting, and Indexing Exercises:"""

"""1. Filter and Select:
        Load the csv_data into a DataFrame.
        Problem: Filter the DataFrame to include only employees in the 'IT' department. 
        From this filtered data, select only the 'name' and 'salary' columns.
        Print the resulting DataFrame."""

df = pd.read_csv(io.StringIO(csv_data))
print("Default DataFrame")
print(df)

it_department = df[df["department"] == "IT"]
print("Filtered DataFrame to include only employees in the 'IT' department. ")
print(it_department)

name_salary_result = it_department[["name", "salary"]]
print("selected only the 'name' and 'salary' columns from IT department")
print(name_salary_result)


"""2. Filter and Sort:

    Load the csv_data into a DataFrame.
    Problem: Filter the DataFrame to include only employees hired in or after the year 2020. 
    Sort this filtered data by 'hire_date' in ascending order.
    Print the resulting DataFrame.
    Hint: You might need to think about how to work with the 'hire_date' column. 
    For simplicity in filtering by year,
    you could potentially treat it as a string for comparison in this exercise,
    or if you're feeling adventurous, look up how to convert it to a datetime object."""


df_dates = pd.read_csv(io.StringIO(csv_data), parse_dates=["hire_date"])
print("DF with parsed dates")
print(df_dates)

hired_2020 = df_dates[df_dates["hire_date"] >= "2020-01-01"]
print("employees hired in or after the year 2020")
print(hired_2020)


hired_sorted = hired_2020.sort_values(by="hire_date")
print("Sorted filtered data by 'hire_date' in ascending order.")
print(hired_sorted)

"""3. Sort and Select (Top N):

    Load the csv_data into a DataFrame.
    Problem: Sort the entire DataFrame by 'salary' in descending order. 
    From this sorted DataFrame, select the top 3 employees (the 3 with the highest salaries)
    using integer position indexing (.iloc[]).
    Print the resulting DataFrame."""

sorted_des = df.sort_values(by="salary", ascending=False)
print("Sorted DataFrame by 'salary' in descending order")
print(sorted_des)

top_3 = sorted_des.iloc[0:3]
print("The top 3 employees (the 3 with the highest salaries)")
print(top_3)


"""4. Combine Filtering, Sorting, and Indexing:

    Load the csv_data into a DataFrame.
    Problem: Filter the DataFrame to include only employees in the 'Sales' or 'Marketing' departments. 
    Sort this filtered data by 'salary' in descending order. 
    From this sorted and filtered data, select the row corresponding to the employee with the highest salary 
    using .loc[] 
    (assuming their original employee_id is still accessible or they retain their original index if you didn't set index_col). 
    If you set employee_id as the index when loading, use .loc[] with the appropriate index label. 
    If you used the default index, 
    you might need to rethink how to get the single highest salary row after sorting and filtering, 
    perhaps by selecting the first row of the result.
    Print the resulting row (which will be a Pandas Series)."""

marketing_sales = df[(df["department"] == "Sales") | (df["department"] == "Marketing")]
print("DataFrame with only employees in the 'Sales' or 'Marketing' departments.")
print(marketing_sales)

sorted_marketing_sales = marketing_sales.sort_values(by="salary", ascending=False)
print("Sorted Marketing and Sales data by 'salary' in descending order.")
print(sorted_marketing_sales)

highest_salary = sorted_marketing_sales.head(1)
print(
    "row corresponding to the employee with the highest salary using .head instead of loc[]"
)
print(highest_salary)

"""
To get the higest salary after sorting and filtering using loc[0] wouldn't work as needed it would return the employee at index 0.
We can manually check the index of the highest employee and use loc[4] but for automating tasks probably this isn't the best solution.
So using .head(1) to get highest value looks better for me, I will discuss this with Gemini to clarify the best answer maybe
loc[] is better but I couldn't handle it correctly.
"""
