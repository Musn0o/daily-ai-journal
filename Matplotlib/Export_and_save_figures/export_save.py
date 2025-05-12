import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Data for Line Plot
days = np.arange(1, 8)
temperatures = [22, 24, 25, 23, 26, 28, 27]

# Data for Bar Plot
classes = ["Math", "Science", "Art", "History"]
student_counts = [30, 25, 15, 20]

# Data for Scatter Plot (with two groups)
hours_studied_group1 = [2, 5, 1, 6, 3]
exam_scores_group1 = [65, 85, 50, 90, 75]

hours_studied_group2 = [4, 7, 6, 8, 5]
exam_scores_group2 = [70, 95, 88, 92, 80]

"""Export and Save Figures Exercises:"""

"""1. Save Customized Line Plot as PNG:

Problem: Recreate the customized line plot showing the weekly temperature variation 
(with title and axis labels). Save this plot as a PNG file named temperature_trend.png.
Write the code to create the plot and save it. 
Check your script's directory to confirm the file is saved."""


fig, ax = plt.subplots()
ax.plot(days, temperatures)
ax.set_title("Temperature Trend")
ax.set_xlabel("Day")
ax.set_ylabel("Temperature")

plt.savefig("Export_and_save_figures/temperature_trend.png")

plt.show()


"""2. Save Customized Scatter Plot as PDF with Options:

    Problem: Recreate the customized scatter plot showing Exam Score vs.
    Hours Studied by Group (with title, axis labels, 
    different colored points for each group, and a legend). 
    Save this plot as a PDF file named exam_score_scatter.pdf. 
    Use dpi=300 for higher resolution and bbox_inches='tight' to remove extra whitespace.
    Write the code to create the plot and save it with these options. 
    Check your script's directory to confirm the file is saved correctly."""

fig, ax = plt.subplots()

ax.scatter(
    exam_scores_group1,
    hours_studied_group1,
    label="Group 1",
    color="blue",
)
ax.scatter(
    exam_scores_group2,
    hours_studied_group2,
    label="Group 2",
    color="red",
)
ax.set_title("Student's study analysis")
ax.set_xlabel("Score")
ax.set_ylabel("Hours")
ax.legend()

plt.savefig(
    "Export_and_save_figures/exam_score_scatter.pdf", dpi=300, bbox_inches="tight"
)
plt.show()


"""3. Save Customized Bar Plot as SVG:

    Problem: Recreate the customized bar plot showing Student Count per Class 
    (with title, axis labels, and colored bars). 
    Save this plot as an SVG file named student_count_bar.svg. 
    SVG is great for graphics that need to scale without losing quality.
    Write the code to create the plot and save it as an SVG. 
    Check your script's directory to confirm the file is saved."""

fig, ax = plt.subplots()

ax.bar(classes, student_counts, color="purple")
ax.set_title("Student Count analysis")
ax.set_xlabel("Class")
ax.set_ylabel("Count")

plt.savefig("Export_and_save_figures/student_count_bar.svg")

plt.show()
