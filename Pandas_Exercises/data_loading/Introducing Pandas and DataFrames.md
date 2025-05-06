Before we load anything, let's quickly introduce Pandas. Pandas is a powerful Python library built on top of NumPy, designed specifically for working with structured data. Its main data structure is the **DataFrame**.

Think of a DataFrame as a table, similar to a spreadsheet in Excel or a table in a database. It has rows and columns, and each column can hold data of a different type (like numbers, text, dates, etc.). This makes it super convenient for organizing and manipulating data!

**Why Load Data?**

Most of the time, the data you'll work with won't be typed directly into your script. It will come from external files like:

- **CSV (Comma Separated Values):** A very common plain text format where values are separated by commas (or other delimiters).
- **Excel (.xlsx, .xls):** Spreadsheet files.
- **JSON (JavaScript Object Notation):** A lightweight data-interchange format that's easy for humans to read and write and easy for machines to parse and generate.
```Python
import pandas as pd

# Assuming you have a file named 'my_data.csv' in the same directory
df = pd.read_csv('my_data.csv')

# Now 'df' is a DataFrame containing the data from the CSV file
print(df)
```

`pd.read_csv()` is quite powerful and has many parameters to handle different scenarios:

- `filepath_or_buffer`: The path to your CSV file (can be a local file path or a URL).
- `sep`: The delimiter used in the file (default is comma ',', but can be tab '\t', semicolon ';', etc.).
- `header`: Row number to use as the column names (default is 0, the first row). Can be set to `None` if your file doesn't have a header row.
- `index_col`: Column(s) to use as the row labels of the DataFrame.

**Example: Reading from a CSV (using a string for demonstration)**

Since we don't have a physical CSV file right now, let's create some sample CSV data as a string and read it using `io.StringIO` (a way to treat a string like a file).

```python
import pandas as pd
import io # We need this to read from a string as if it were a file

csv_data = """name,age,city
Alice,30,New York
Bob,25,London
Charlie,35,Paris"""

# Use io.StringIO to read the string data
df_csv = pd.read_csv(io.StringIO(csv_data))

print("DataFrame loaded from CSV data:")
print(df_csv)
```

**Loading Data from Excel Files (`pd.read_excel()`)**

Reading Excel files is also straightforward using `pd.read_excel()`.

```python
# Assuming you have a file named 'my_excel_data.xlsx'
# You might need to install the openpyxl library for .xlsx files: pip install openpyxl
# Or xlrd for older .xls files: pip install xlrd
df_excel = pd.read_excel('my_excel_data.xlsx', sheet_name='Sheet1') 
# Specify sheet name if needed
```

`pd.read_excel()` is similar to `pd.read_csv()` but has parameters specific to Excel, like `sheet_name` to specify which sheet to read from.

**Loading Data from JSON Files (`pd.read_json()`)**

If your data is in JSON format, `pd.read_json()` is your friend.

```python
# Assuming you have a file named 'my_json_data.json'
df_json = pd.read_json('my_json_data.json')
```

The structure of your JSON data will determine how `pd.read_json()` interprets it into a DataFrame.

**Which format to use?**

- **CSV:** Great for simple tabular data, easy to work with in text editors, widely compatible.
- **Excel:** Useful when data is already in spreadsheet format, can handle multiple sheets.
- **JSON:** Good for semi-structured or hierarchical data, commonly used in web APIs.

For this subtask, our main goal is to get comfortable using these `pd.read_...` functions.

Ready for a quick exercise to practice loading some data into a Pandas DataFrame? 😉 We can start with a simple CSV!


**Exercise 2 (CSV string with a different delimiter):** Ah, the "shifted to the right" observation! 🤔 That's a classic sign that Pandas didn't correctly identify the delimiter. By default, `pd.read_csv()` expects commas. If your data uses semicolons (`;`) but you don't tell `pd.read_csv()` about it, it will try to read the _entire_ line as a single value in the first column (or maybe try to guess incorrectly), causing everything to look shifted.

The key to fixing this is using the `sep` parameter:

```python
df_semicolon = pd.read_csv(io.StringIO(csv_data_2), sep=';')
```

Adding `sep=';'` tells Pandas to use the semicolon as the separator between values. This should correctly parse the data into three distinct columns: `product_id`, `product_name`, and `price`.