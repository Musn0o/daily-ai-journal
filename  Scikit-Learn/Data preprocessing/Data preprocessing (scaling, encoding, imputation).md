This step is incredibly important in the machine learning pipeline. Raw data rarely comes in a format that's immediately suitable for machine learning models. Preprocessing involves transforming your data to make it compatible with ML algorithms and often improves model performance.

We'll cover three key techniques here:

1. **Imputation:** Handling Missing Values
2. **Encoding:** Handling Categorical Data
3. **Scaling:** Handling Numerical Data

Let's use an updated version of our employee DataFrame that includes some missing values and a new categorical column to demonstrate these techniques:

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Updated Employee DataFrame with more varied data and a new categorical column 'education_level'
data = {'employee_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'department': ['Sales', 'IT', 'Sales', 'IT', 'Marketing', 'IT', 'Sales', 'Marketing', 'IT', 'Sales'],
        'salary': [60000, 75000, 65000, 80000, 70000, 78000, 62000, 72000, 79000, np.nan],
        'years_experience': [2, 5, 3, 7, 1, 4, 2, 3, 6, 1],
        'performance_score': [85, 90, 88, 92, np.nan, 91, 86, 89, 93, 80], # Added NaN here
        'education_level': ['Bachelor', 'Master', 'PhD', 'Master', 'Bachelor', 'PhD', 'Bachelor', 'Master', 'PhD', 'Bachelor']} # New categorical column

df = pd.DataFrame(data)
print("Original DataFrame:\n", df)

# Let's define our features (X) and target (y) for potential ML modeling later
# For preprocessing, we often work on features separately
X = df[['years_experience', 'performance_score', 'department', 'education_level']]
y = df['salary'] # We'll need to handle NaN in y later too
```

### 1. Imputation: Handling Missing Values

Machine learning models generally cannot handle `np.nan` (missing values). You either drop rows/columns with NaNs (which you know from Pandas `.dropna()`), or you **impute** them, meaning you fill them in with a calculated value.

Scikit-learn's `SimpleImputer` is commonly used for this.

- **When to use:** When you have `NaN` values in your numerical or categorical columns.
- **Strategy:** Common strategies include:
    - `'mean'`: Fill with the mean of the column (for numerical data).
    - `'median'`: Fill with the median of the column (for numerical data, robust to outliers).
    - `'most_frequent'`: Fill with the most common value (for numerical or categorical data).
    - `'constant'`: Fill with a specified constant value.

```python
# --- Imputation Example ---
print("\n--- Imputation ---")

# Let's focus on 'performance_score' which has a NaN
numerical_data_for_imputation = X[['years_experience', 'performance_score']]

# Create an imputer that fills missing values with the mean
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the data (it learns the mean)
imputer.fit(numerical_data_for_imputation)

# Transform the data (fill the NaNs)
X_imputed_numerical = imputer.transform(numerical_data_for_imputation)

print("Original 'performance_score' column:\n", df['performance_score'])
print("\nImputed numerical data (Numpy array):\n", X_imputed_numerical)
# You often convert back to DataFrame if you want to keep column names
X_imputed_df = pd.DataFrame(X_imputed_numerical, columns=numerical_data_for_imputation.columns)
print("\nImputed numerical DataFrame:\n", X_imputed_df)
```

**Important Note on Imputation:** If you have `NaN` in your target `y` column, you generally either drop those rows or impute them based on your specific problem. For the `salary` column (`y` in our example), dropping NaNs (as we did in the previous exercise with `df.dropna(subset=['salary'])`) is often the most straightforward approach if you can afford to lose the rows.

---

### 2. Encoding: Handling Categorical Data

Machine learning models primarily work with numbers. You can't directly feed text categories like 'Sales' or 'Bachelor' into most algorithms. You need to convert them into numerical representations.

- **When to use:** When you have categorical (textual) features.

Two main types of encoding:

- **a) One-Hot Encoding (`OneHotEncoder`): For Nominal (Unordered) Categories**
    
    - **Purpose:** Used when categories have no intrinsic order (e.g., 'Red', 'Blue', 'Green'; 'Sales', 'IT', 'Marketing').
    - **How it works:** Creates a new binary (0 or 1) column for each unique category. A `1` indicates the presence of that category, and `0` indicates its absence.
    - **Advantage:** Prevents the model from incorrectly assuming an order between categories.
    - **Disadvantage:** Can lead to a high number of new columns (and increase dataset size) if you have many unique categories.

```python
# --- One-Hot Encoding Example ---
print("\n--- One-Hot Encoding ---")

# Let's focus on the 'department' column
categorical_nominal_data = X[['department']]

# Create a OneHotEncoder instance
encoder_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
# handle_unknown='ignore' handles categories not seen during fit
# sparse_output=False returns a dense numpy array (easier to see)

# Fit and transform the data
X_encoded_department = encoder_ohe.fit_transform(categorical_nominal_data)

print("Original 'department' column:\n", df['department'])
print("\nOne-Hot Encoded 'department' (Numpy array):\n", X_encoded_department)
print("Encoded feature names:", encoder_ohe.get_feature_names_out(['department']))
# Convert to DataFrame to see columns clearly
X_encoded_department_df = pd.DataFrame(X_encoded_department, columns=encoder_ohe.get_feature_names_out(['department']))
print("\nOne-Hot Encoded DataFrame:\n", X_encoded_department_df)
```

**b) Ordinal Encoding (`OrdinalEncoder`): For Ordinal (Ordered) Categories**

- **Purpose:** Used when categories have a natural order (e.g., 'Low', 'Medium', 'High'; 'Bachelor', 'Master', 'PhD').
- **How it works:** Assigns a unique integer to each category based on its order.
- **Advantage:** Fewer new columns than One-Hot Encoding.
- **Caution:** Only use if a clear order exists; otherwise, the model might misinterpret the numerical relationship.

```python
# --- Ordinal Encoding Example ---
print("\n--- Ordinal Encoding ---")

# Let's focus on the 'education_level' column, assuming an order
categorical_ordinal_data = X[['education_level']]

# Create an OrdinalEncoder instance, specify the order
# This is crucial for ordinal encoding!
education_order = ['Bachelor', 'Master', 'PhD']
encoder_ord = OrdinalEncoder(categories=[education_order])

# Fit and transform the data
X_encoded_education = encoder_ord.fit_transform(categorical_ordinal_data)

print("Original 'education_level' column:\n", df['education_level'])
print("\nOrdinal Encoded 'education_level' (Numpy array):\n", X_encoded_education)
# Convert to DataFrame for readability
X_encoded_education_df = pd.DataFrame(X_encoded_education, columns=categorical_ordinal_data.columns)
print("\nOrdinal Encoded DataFrame:\n", X_encoded_education_df)
```

### 3. Scaling: Handling Numerical Data

Many machine learning algorithms are sensitive to the scale of numerical features. If one feature has values ranging from 0 to 1000 and another from 0 to 1, the model might implicitly give more weight to the larger-scaled feature. Scaling ensures all features contribute equally.

- **When to use:** For numerical features, especially for distance-based algorithms (like K-Nearest Neighbors, K-Means, SVMs) or those using gradient descent optimization (like Linear Regression, Neural Networks).
- **Purpose:** To bring all numerical features to a similar scale without distorting differences in the ranges of values or losing information.

Two common types of scaling:

- **a) Standardization (`StandardScaler`): Z-score normalization**
    
    - **Purpose:** Transforms data to have a mean of 0 and a standard deviation of 1.
    - **Formula:** $xscaled​=\frac{x−μ}{σ}$​ (where μ is the mean and σ is the standard deviation).
    - **When to use:** Good choice when your data follows a Gaussian (bell curve) distribution, or when your algorithm assumes normally distributed data.

```python
# --- Standardization Example ---
print("\n--- Standardization (StandardScaler) ---")

# Let's use the imputed numerical data (or original if no NaNs)
# For this example, let's just use 'years_experience' for simplicity
numerical_data_for_scaling = X[['years_experience']]

# Create a StandardScaler instance
scaler_std = StandardScaler()

# Fit and transform the data
X_scaled_std = scaler_std.fit_transform(numerical_data_for_scaling)

print("Original 'years_experience' column:\n", df['years_experience'])
print("\nStandardized 'years_experience' (Numpy array):\n", X_scaled_std)
```

**b) Normalization (`MinMaxScaler`): Min-Max Scaling**

- **Purpose:** Transforms data to a fixed range, typically between 0 and 1.
- **Formula:** $xscaled​=\frac{x−min(x)}{max(x)−min(x)​}$
- **When to use:** Useful when you need features to be within a specific range, or for algorithms that are sensitive to the magnitude of the features (like neural networks).

```python
# --- Normalization (MinMaxScaler) Example ---
print("\n--- Normalization (MinMaxScaler) ---")

# Using the same numerical data for consistency
numerical_data_for_scaling = X[['years_experience']]

# Create a MinMaxScaler instance
scaler_minmax = MinMaxScaler()

# Fit and transform the data
X_scaled_minmax = scaler_minmax.fit_transform(numerical_data_for_scaling)

print("Original 'years_experience' column:\n", df['years_experience'])
print("\nNormalized 'years_experience' (Numpy array):\n", X_scaled_minmax)
```

**Crucial Point: `fit()` only on Training Data!**

Remember our discussion about train-test splitting? When doing preprocessing, you must always:

1. `fit()` the imputer, encoder, or scaler **only on the training data**.
2. Then `transform()` **both** the training data AND the test data using the _same_ fitted transformer. This prevents "data leakage" from the test set into your preprocessing steps. Scikit-learn transformers have a `fit_transform()` method for convenience, which `fit()`s and then `transform()`s in one step, often used on the training set.

```python
# Example of fit_transform on training and transform on test
# Let's simulate a split first
X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(
    X[['years_experience']], y, test_size=0.3, random_state=42
) # Using a simplified X for demonstration

scaler_std_correct = StandardScaler()

# Fit on training data AND transform training data
X_train_scaled = scaler_std_correct.fit_transform(X_temp_train)

# ONLY transform test data (using the mean/std learned from training data)
X_test_scaled = scaler_std_correct.transform(X_temp_test)

print("\n--- Correct Scaling with Train-Test Split ---")
print("X_train_scaled (first 5):\n", X_train_scaled[:5])
print("X_test_scaled (first 5):\n", X_test_scaled[:5])
```

That's a lot of information, but these are the workhorses of data preparation!

How does this introduction to imputation, encoding, and scaling feel? Ready to apply these techniques in some exercises? 😉 Let's make that data model-ready! 💪