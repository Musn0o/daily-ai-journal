import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn


"""Seaborn Heatmap and Pair Plot Exercises:"""

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
    "salary": [60000, 75000, 65000, 80000, 70000, 78000, 62000, 72000, 79000, np.nan],
    "years_experience": [2, 5, 3, 7, 1, 4, 2, 3, 6, 1],
    "performance_score": [85, 90, 88, 92, 87, 91, 86, 89, 93, 80],
}  # Added performance_score

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)


"""1. Create a Correlation Heatmap:

    Problem: Calculate the correlation matrix for the numerical columns in the df DataFrame. 
    Create a heatmap of this correlation matrix using sns.heatmap().
    Make sure the correlation values are annotated on the heatmap (annot=True).
    Choose a colormap (e.g., 'viridis', 'plasma', 'coolwarm').
    Add a title to the heatmap (remember you might need Matplotlib for this!).
    Print the correlation matrix before displaying the heatmap.
    Write the code to create and display this heatmap."""

df = df.fillna(0)
correlation = df.corr(numeric_only=True)
print("correlation matrix")
print(correlation)

fig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=ax)

ax.set_title("Scar's Correlation Matrix Heatmap")

plt.show()

"""2. Create a Pair Plot:

    Problem: Create a pair plot for the df DataFrame using sns.pairplot().
    Use the 'department' column as the hue to color the points based on the department.
    Write the code to create and display this pair plot."""

sns.pairplot(df, hue="department")

plt.show()
