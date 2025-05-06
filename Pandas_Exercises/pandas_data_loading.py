import pandas as pd
import io


"""Pandas Data Loading Exercises:

    1- Load Data from a CSV String:"""

csv_data_1 = """
student_id,name,major
1,Alice,Computer Science
2,Bob,Physics
3,Charlie,Chemistry
"""

df_csv1 = pd.read_csv(io.StringIO(csv_data_1))

print("DataFrame loaded from CSV data 1:")
print(df_csv1)


"""2- Load Data from a CSV String with a Different Delimiter:"""

csv_data_2 = """
product_id;product_name;price
101;Laptop;1200
102;Keyboard;75
103;Mouse;25
"""

df_csv2 = pd.read_csv(io.StringIO(csv_data_2), sep=";")

print("DataFrame loaded from CSV data 2:")
print(df_csv2)


"""3-Load Data from a JSON String:"""

json_data_1 = """
[
  {"book_id": "001", "title": "The Great Novel", "author": "Author A"},
  {"book_id": "002", "title": "Another Story", "author": "Author B"},
  {"book_id": "003", "title": "Mystery Solved", "author": "Author C"}
]
"""

df_json = pd.read_json(io.StringIO(json_data_1))

print("DataFrame loaded from JSON data 1:")
print(df_json)


"""4- (Think and Discuss - No coding required):

    Imagine you have an actual CSV file named my_local_data.csv saved on your computer
    in the same folder where your Python script is. How would you modify the code
    from Exercise 1 to read data directly from this file instead of the string?"""

"""The difference won't be big
I would only use this line df_csv1 = pd.read_csv("Pandas_Exercises/my_data.csv")
assuming the file name is my_data.csv actually this looks much easier than using StringIO
 """
