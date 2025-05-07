import pandas as pd
import numpy as np

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

""".fillna() and Fill Methods Exercise:"""

"""1. Fill with a scalar: 
Create a new DataFrame df_filled_scalar where all NaN values are replaced with the value -99. 
Print df_filled_scalar."""

df_filled_scalar = df.fillna(-99)
print("df_filled_scalar where all NaN values are replaced with the value -99")
print(df_filled_scalar)

"""2. Fill with column mean: 
Create a new DataFrame df_filled_mean where the NaN values in column 'A' are replaced
with the mean of column 'A', and the NaN values in column 'B' are replaced with the mean of column 'B'.
(Hint: You can calculate the means first and then use the dictionary approach with fillna). 
Print df_filled_mean."""

df_mean_a = df["A"].mean()
df_mean_b = df["B"].mean()
print(df_mean_a, df_mean_b)

filling_dict = {"A": df_mean_a, "B": df_mean_b}
df_filled_mean = df.fillna(value=filling_dict)
print("df_filled_mean where the A & B NaN values are replaced by mean")
print(df_filled_mean)


"""3. Forward Fill: 
Create a new DataFrame df_ffilled using the forward fill method (.ffill()) to fill NaN values. 
Print df_ffilled."""

df_ffilled = df.ffill()

print("df_ffilled using the forward fill method (.ffill()) to fill NaN values")
print(df_ffilled)

"""4. Backward Fill: 
Create a new DataFrame df_bfilled using the backward fill method (.bfill()) to fill NaN values. 
Print df_bfilled."""

df_bfilled = df.bfill()
print("df_bfilled using the backward fill method (.bfill()) to fill NaN values")
print(df_bfilled)

"""5. Forward Fill with Limit: 
Create a new DataFrame df_ffill_limit using the forward fill method (.ffill()), 
but only fill a maximum of 1 consecutive NaN value at a time. Print df_ffill_limit."""

df_ffill_limit = df.ffill(limit=1)
print("df_ffill_limit using the forward fill method but only fill a maximum of 1")
print(df_ffill_limit)
