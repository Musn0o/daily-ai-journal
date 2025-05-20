import pandas as pd
import numpy as np

"""Exercise:

    Load a CSV dataset (e.g., IMDB reviews) into a Pandas DataFrame.
    Clean the data: remove duplicates, handle missing values, and add a column for text length.
    Use NumPy to compute the average length of reviews.
"""
# Load the dataset
df = pd.read_csv("np_pd_exercise/IMDB Dataset.csv")

# Clean the data
# Remove duplicates
df.drop_duplicates(inplace=True)
# Handle missing values
df.ffill(inplace=True)  # Forward fill missing values
# Add a column for text length
df["text_length"] = df["review"].str.len()
# Compute the average length of reviews
average_length = np.mean(df["text_length"])
print(f"The average length of reviews is: {average_length:.2f}")
