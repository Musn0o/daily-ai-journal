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
"""We can also use this df_dates['hire_date'].str.startswith('202')"""


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

# Alternative
# highest_salary = sorted_marketing_sales.iloc[0]

print(
    "row corresponding to the employee with the highest salary using .head instead of loc[]"
)
print(highest_salary)

"""
Scar:
To get the higest salary after sorting and filtering using loc[0] wouldn't work as needed it would return the employee at index 0.
We can manually check the index of the highest employee and use loc[4] but for automating tasks probably this isn't the best solution.
So using .head(1) to get highest value looks better for me, I will discuss this with Gemini to clarify the best answer maybe
loc[] is better but I couldn't handle it correctly.
"""

"""
Gemini: 
The issue with using .loc[] here is that you need to know the specific index label of the row with the highest salary in the filtered and sorted DataFrame.
Since you sorted by salary descending, the row with the highest salary in the filtered result will always be the first row (at integer position 0) of that filtered and sorted DataFrame.
Therefore, the most straightforward and robust way to get that single row after filtering and sorting by salary is to use .iloc[0] on the filtered and sorted DataFrame. This selects the row at integer position 0, which is guaranteed to be the row with the highest salary after you've sorted by salary descending.
Alternatively, if you really wanted to use .loc[], you would first need to get the index label of the first row of your filtered and sorted DataFrame (e.g., filtered_sorted_df.index[0]) and then use that label with .loc[]. 
But .iloc[0] is much more direct in this specific scenario.
"""

"""
Scar:
 Before I close the sorting, filtering and indexing note. 
 I need to confirm which method would be better for the last exercise, using ".head()" or "iloc[0]". 
 Or there is no better solution, they are both the same in this situation? 
"""

"""
Gemini:
That's a fantastic question to refine your understanding! 🤔 It highlights the subtle differences between methods that might produce similar results in specific situations.

You're right, in the case of getting the single top row after sorting, both .head(1) and .iloc[0] will give you the data of the employee with the highest salary. However, there is a key difference in what they return and their typical use case:

    .head(n):
        This method is primarily used for previewing the top n rows of a DataFrame.
        It always returns a DataFrame, even if you ask for only one row (.head(1) will return a DataFrame with one row).

    .iloc[position]:
        This indexer is used for selecting data by integer position.
        When you use .iloc[0] on a DataFrame, you are specifically asking for the element at integer position 0, which is the first row.
        .iloc[0] on a DataFrame returns a Pandas Series, which represents that single row.

Which is "better" in this scenario?

There isn't a strict "better" in terms of correctness, as both will give you the data for the highest earner. However, based on typical usage and what you might want to do next with the result:

    If your goal is simply to display or inspect the top row, .head(1) is perfectly fine and clear.
    If you intend to then access individual elements within that top row (e.g., get the name, get the salary), using .iloc[0] which returns a Series might be slightly more convenient because you can then access the values directly like highest_earner_series['name']. If you used .head(1), you'd get a DataFrame, and you'd need another step to access the values (e.g., highest_earner_df['name'].iloc[0]).

For selecting a specific single item by its integer position, .iloc[0] is often considered more direct and idiomatic in Pandas. It explicitly says "give me the item at position 0".

So, while both work, .iloc[0] is often preferred when you specifically want that single first row as a Series for further processing, while .head(1) is more for just quickly looking at the top of the DataFrame.

Does that distinction make sense? It's a subtle but important difference in how Pandas methods and indexers return data! 😊 You're asking great questions that really help solidify your understanding! 👍
"""
