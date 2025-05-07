import pandas as pd
import numpy as np


"""Let's get some practice with .dropna()! 💪"""

data = {
    "A": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan],
    "B": [5.0, np.nan, np.nan, 8.0, 9.0, 10.0],
    "C": [9.0, 10.0, 11.0, 12.0, np.nan, 14.0],
    "D": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    "E": [15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
}

df = pd.DataFrame(data)
print("Original DataFrame with missing data:")
print(df)

""".dropna() Exercise:"""

"""1. Drop rows with any missing value: 
Create a new DataFrame df_cleaned_1 by dropping any row that contains at least one NaN. 
Print df_cleaned_1.
"""

df_cleaned_1 = df.dropna()
print("dropping any row that contains at least one NaN")
print(df_cleaned_1)

"""Empty DF because all rows had NaN"""

"""2. Drop columns with any missing value: 
Create a new DataFrame df_cleaned_2 by dropping any column that contains at least one NaN. 
Print df_cleaned_2."""

df_cleaned_2 = df.dropna(axis=1)
print("dropping any column that contains at least one NaN")
print(df_cleaned_2)

"""Only E column didn't have any NaN values"""

"""3. Drop rows where all values are missing: 
Create a new DataFrame df_cleaned_3 by dropping only the rows where all values are NaN. 
Print df_cleaned_3."""

df_cleaned_3 = df.dropna(how="all")
print("dropping only the rows where all values are NaN")
print(df_cleaned_3)

"""No rows were dropped because none of them has full NaN values"""

"""4. Drop columns where all values are missing: 
Create a new DataFrame df_cleaned_4 by dropping only the columns where all values are NaN. 
Print df_cleaned_4."""

df_cleaned_4 = df.dropna(axis="columns", how="all")
print("dropping only the columns where all values are NaN")
print(df_cleaned_4)

"""The D column were removed because it was the one who had all NaN"""

"""5. Drop rows based on a threshold: 
Create a new DataFrame df_cleaned_5 by dropping rows that have less than 4 non-NaN values. Print df_cleaned_5."""

df_cleaned_5 = df.dropna(axis="index", thresh=4)
print("dropping rows that have less than 4 non-NaN values")
print(df_cleaned_5)

"""Only index 0 & 3 has 4 non-NaN values so other rows(index) were removed by our thresh"""
