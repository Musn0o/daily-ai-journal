Our first subtask in Data Visualization is **"Create line, bar, and scatter plots"**. We'll focus on using **Matplotlib** for these fundamental plot types.

**Introducing Matplotlib**

Matplotlib is a powerful and flexible plotting library. At its core, a Matplotlib plot consists of:

- **Figure:** This is the overall window or page that contains the plot(s). You can think of it as the canvas.
- **Axes:** This is the actual area where the data is plotted. A Figure can contain one or more Axes. When you create a single plot, you usually have one Figure and one set of Axes.

We'll primarily use the **object-oriented interface** in Matplotlib, which gives you more control by explicitly creating Figure and Axes objects.

To get started, you typically import the `pyplot` module, which is the most commonly used part of Matplotlib:

```python
import matplotlib.pyplot as plt
import pandas as pd # We'll often plot data from Pandas DataFrames
```

**Creating a Basic Plot Structure**

The standard way to create a Figure and Axes is using `plt.subplots()`:

```python
# Create a Figure and an Axes object
fig, ax = plt.subplots()

# Now you use the 'ax' object to plot your data
# ax.plot(...)
# ax.bar(...)
# ax.scatter(...)

# Finally, display the plot
plt.show()
```

`plt.subplots()` is convenient because it creates both the Figure and a set of Axes for you at once.

**Let's look at the basic plot types:**

**1. Line Plots (`ax.plot()`):**

- **When to use:** Ideal for showing trends over time or sequential data. It connects data points with lines.
- **Example:** Plotting a simple trend.

```python
# Example Data
x_data = [1, 2, 3, 4, 5]
y_data = [2, 4, 5, 4, 6]

# Create Figure and Axes
fig, ax = plt.subplots()

# Create a line plot
ax.plot(x_data, y_data)

# Display the plot
plt.show()
```

**2. Bar Plots (`ax.bar()`):**

- **When to use:** Great for comparing discrete categories. The height of each bar represents the value for that category.
- **Example:** Comparing the count or value for different categories.

```python
# Example Data
categories = ['A', 'B', 'C', 'D']
values = [10, 25, 15, 30]

# Create Figure and Axes
fig, ax = plt.subplots()

# Create a bar plot
ax.bar(categories, values)

# Display the plot
plt.show()
```

**3. Scatter Plots (`ax.scatter()`):**

- **When to use:** Used to show the relationship between two numerical variables. Each point on the plot represents an observation. Useful for spotting correlations or clusters.
- **Example:** Plotting the relationship between two numerical features.

```python
# Example Data
x_scatter = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11]
y_scatter = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78]

# Create Figure and Axes
fig, ax = plt.subplots()

# Create a scatter plot
ax.scatter(x_scatter, y_scatter)

# Display the plot
plt.show()
```

These are the basic ways to create these fundamental plot types using Matplotlib's object-oriented interface. The key is to create your Figure and Axes first and then use the plotting methods (`.plot()`, `.bar()`, `.scatter()`) on the `ax` object.

How does that introduction to creating basic plots with Matplotlib feel? Does the Figure and Axes concept make sense? 😊

Ready to try some exercises where you create these plots using some sample data? 😉 Let's make some visualizations! ✨