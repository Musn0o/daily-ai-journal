import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


"""Matplotlib Customization Exercises:"""

"""1. Customize a Line Plot:

    Problem: Create a line plot using the days and temperatures data. Add the following:
        Title: "Weekly Temperature Variation"
        X-axis label: "Day"
        Y-axis label: "Temperature (°C)"
    Write the code to create and display this customized line plot."""

fig, ax = plt.subplots()

ax.plot(days, temperatures)
ax.set_title("Weekly Temperature Variation")
ax.set_xlabel("Day")
ax.set_ylabel("Temperature (°C)")

plt.show()


"""2. Customize a Bar Plot:

    Problem: Create a bar plot using the classes and student_counts data. Add the following:
        Title: "Student Count per Class"
        X-axis label: "Class"
        Y-axis label: "Number of Students"
        Change the color of the bars (choose any color you like!).
    Write the code to create and display this customized bar plot."""

fig, ax = plt.subplots()

ax.bar(classes, student_counts, color="black")
ax.set_title("Student Count per Class")
ax.set_xlabel("Class")
ax.set_ylabel("Number of Students")

plt.show()

"""3. Customize a Scatter Plot with Legend:

    Problem: Create a scatter plot showing the relationship between hours studied and exam scores, 
    using both group1 and group2 data. Add the following:
        Title: "Exam Score vs. Hours Studied by Group"
        X-axis label: "Hours Studied"
        Y-axis label: "Exam Score"
        Plot the points for group1 in one color (e.g., blue) and label them "Group 1".
        Plot the points for group2 in a different color (e.g., red) and label them "Group 2".
        Add a legend to distinguish the two groups.
    Write the code to create and display this customized scatter plot."""

fig, ax = plt.subplots()

ax.scatter(hours_studied_group1, exam_scores_group1, label="Group 1", color="blue")
ax.scatter(hours_studied_group2, exam_scores_group2, label="Group 2", color="red")

ax.set_title("Exam Score vs. Hours Studied by Group")
ax.set_xlabel("Hours Studied")
ax.set_ylabel("Exam Score")

ax.legend()
plt.show()
