import matplotlib.pyplot as plt
import pandas as pd  # We'll often plot data from Pandas DataFrames


"""Basic Matplotlib Plotting Exercises:"""

"""1. Create a Line Plot:

Here is some hypothetical data representing temperature over 7 days:

days = [1, 2, 3, 4, 5, 6, 7]
temperatures = [22, 24, 25, 23, 26, 28, 27]

Problem: Create a line plot showing the trend of temperature over these 7 days.
Write the code to create and display this line plot.
"""

days = [1, 2, 3, 4, 5, 6, 7]
temperatures = [22, 24, 25, 23, 26, 28, 27]


fig, ax = plt.subplots()

ax.plot(days, temperatures)

plt.show()


"""2. Create a Bar Plot:

Here is some data representing the number of students in different classes:
    

classes = ['Math', 'Science', 'Art', 'History']
student_counts = [30, 25, 15, 20]

Problem: Create a bar plot comparing the number of students in each class.
Write the code to create and display this bar plot."""

classes = ["Math", "Science", "Art", "History"]
student_counts = [30, 25, 15, 20]

fig, ax = plt.subplots()

ax.bar(classes, student_counts)

plt.show()


"""3.Create a Scatter Plot:

Here is some hypothetical data showing the relationship between hours studied and exam scores for a few students:


hours_studied = [2, 5, 1, 6, 3, 7, 4]
exam_scores = [65, 85, 50, 90, 75, 95, 80]

Problem: Create a scatter plot to visualize the relationship between hours studied and exam scores.
Write the code to create and display this scatter plot."""

hours_studied = [2, 5, 1, 6, 3, 7, 4]
exam_scores = [65, 85, 50, 90, 75, 95, 80]


figo, axe = plt.subplots()

ax.scatter(hours_studied, exam_scores)

plt.show()
