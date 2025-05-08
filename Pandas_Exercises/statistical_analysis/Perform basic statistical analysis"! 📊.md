This is where we use Pandas to quickly calculate descriptive statistics and get a better understanding of our data's characteristics.

Pandas DataFrames and Series come equipped with a wide range of methods for calculating common statistics. These methods ignore `NaN` values by default (which is often the desired behavior).

Let's use a slightly modified version of our employee salary data for these examples:

```python
import pandas as pd
import numpy as np

data = {'employee_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'department': ['Sales', 'IT', 'Sales', 'IT', 'Marketing', 'IT', 'Sales', 'Marketing', 'IT', 'Sales'],
        'salary': [60000, 75000, 65000, 80000, 70000, 78000, 62000, 72000, 79000, np.nan], # Added a NaN salary
        'years_experience': [2, 5, 3, 7, 1, 4, 2, 3, 6, 1]} # Added years of experience

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
```

**Common Statistical Methods:**

1. **`.describe()`:** This is a super handy method that generates a variety of descriptive statistics for numerical columns in your DataFrame (or Series).

```python
# Get descriptive statistics for numerical columns
descriptive_stats = df.describe()
print("\nDescriptive statistics for numerical columns:")
print(descriptive_stats)

# You can also include non-numerical columns by adding include='all'
descriptive_stats_all = df.describe(include='all')
print("\nDescriptive statistics for all columns (includes object types):")
print(descriptive_stats_all)
```

- `.describe()` gives you count (non-NaN), mean, standard deviation (std), minimum (min), maximum (max), and the quartiles (25%, 50% - which is the median, and 75%).
    
2. **Measures of Central Tendency:**
    
    - **`.mean()`:** Calculates the average.
    - **`.median()`:** Calculates the middle value.
    - **`.mode()`:** Calculates the most frequent value(s).

```python
# Mean salary
mean_salary = df['salary'].mean()
print(f"\nMean Salary: {mean_salary:.2f}")

# Median years of experience
median_experience = df['years_experience'].median()
print(f"Median Years of Experience: {median_experience}")

# Mode of department (returns a Series as there could be multiple modes)
mode_department = df['department'].mode()
print("Mode of Department:")
print(mode_department)
```

3. **Measures of Dispersion:**

	- **`.std()`:** Calculates the standard deviation.
	- **`.var()`:** Calculates the variance.

```python
# Standard deviation of salary
std_salary = df['salary'].std()
print(f"\nStandard Deviation of Salary: {std_salary:.2f}")

# Variance of years of experience
var_experience = df['years_experience'].var()
print(f"Variance of Years of Experience: {var_experience:.2f}")
```

4. **Other Useful Aggregations:**

	- **`.sum()`:** Calculates the sum.
	- **`.count()`:** Counts non-NaN values.
	- **`.size()`:** Counts the total number of elements (including NaNs).
	- **`.min()`:** Finds the minimum value.
	- **`.max()`:** Finds the maximum value.

```python
# Total sum of salaries (NaN is ignored)
total_salary_sum = df['salary'].sum()
print(f"\nTotal Sum of Salaries (ignoring NaN): {total_salary_sum:.2f}")

# Count of non-NaN salaries
count_salaries = df['salary'].count()
print(f"Count of non-NaN Salaries: {count_salaries}")

# Total number of rows in the salary column (including NaN)
size_salary_col = df['salary'].size
print(f"Total number of elements in Salary column (including NaN): {size_salary_col}")

# Minimum salary
min_salary = df['salary'].min()
print(f"Minimum Salary: {min_salary}")

# Maximum years of experience
max_experience = df['years_experience'].max()
print(f"Maximum Years of Experience: {max_experience}")
```

5. **Quantiles/Percentiles (`.quantile()`):**

	- Calculates the value at a specific quantile (e.g., the value below which 25% of the data falls).

```python
# 25th percentile of salary
salary_25th_percentile = df['salary'].quantile(0.25)
print(f"\n25th Percentile of Salary: {salary_25th_percentile:.2f}")

# You can get quartiles (25th, 50th, 75th) using a list
salary_quartiles = df['salary'].quantile([0.25, 0.5, 0.75])
print("\nSalary Quartiles:")
print(salary_quartiles)
```

**Statistical Analysis with Grouping:**

Many of these statistical methods can be applied _after_ grouping your data to get statistics for each group, which is incredibly powerful for comparative analysis. We saw this with `.mean()` after grouping by department.

```python
# Get descriptive statistics for salary per department
department_salary_description = df.groupby('department')['salary'].describe()
print("\nDescriptive statistics for salary per department:")
print(department_salary_description)
```

This gives you the count, mean, std, min, max, etc., for salaries within each department!

These are the basic building blocks for performing statistical analysis in Pandas. By combining them with filtering and grouping, you can gain deep insights into your data!

How does performing basic statistical analysis with these Pandas methods feel? Are you ready to try some exercises to calculate statistics on the DataFrame, including after grouping? 😉 Let's put that energy to work! 💪