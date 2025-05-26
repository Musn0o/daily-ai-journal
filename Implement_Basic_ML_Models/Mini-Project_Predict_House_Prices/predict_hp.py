import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingRegressor
import joblib


# Set random seed for reproducibility
np.random.seed(42)

# -------------------- Phase 1: Data Acquisition & Initial Exploration (EDA) --------------------

"""Phase 1: Data Acquisition & Initial Exploration (EDA)
    1.1.Load Data: Load both train.csv and test.csv into pandas DataFrames.

    1.2.Initial Inspection:
        Check df.head(), df.info(), df.describe().
        Identify categorical and numerical features.
        Crucially: Check for and quantify missing values (df.isnull().sum()). This will be a major preprocessing task!
        Basic visualization: Look at the distribution of the target variable (SalePrice) using a histogram.
"""

# 1.1. Load Data: Load both train.csv and test.csv into pandas DataFrames.
train_df = pd.read_csv("Mini-Project_Predict_House_Prices/data/train.csv")
test_df = pd.read_csv("Mini-Project_Predict_House_Prices/data/test.csv")

# Keep a copy of the raw data for reference
train_raw = train_df.copy()
test_raw = test_df.copy()


# Print shape of train and test DataFrames for sanity check
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


def eda_report(df, target=None, name="DataFrame"):
    """
    Perform initial EDA on the given DataFrame.
    Prints head, info, describe, column types, missing values, and target distribution.
    """
    print(f"\n--- {name} HEAD ---")
    print(df.head())

    print(f"\n--- {name} INFO ---")
    print(df.info())

    print(f"\n--- {name} DESCRIBE ---")
    print(df.describe())

    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")

    # Check for and quantify missing values (sorted)
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(f"\nMissing values (sorted):\n{missing}")

    # If target is provided, show its distribution
    if target and target in df.columns:
        print(f"\n--- {target} Distribution ---")
        print(f"Skewness: {df[target].skew():.2f}")
        print(f"Kurtosis: {df[target].kurt():.2f}")
        df[target].hist(bins=30, color="green")
        plt.title(f"{target} Distribution")
        plt.xlabel(target)
        plt.ylabel("Frequency")
        plt.show()

    return categorical_cols, numerical_cols


# Run EDA on train and test sets
train_categorical_cols, train_numerical_cols = eda_report(
    train_df, target="SalePrice", name="Train Data"
)
test_categorical_cols, test_numerical_cols = eda_report(test_df, name="Test Data")

# Store column lists for later use in preprocessing
# (You may want to use set intersection to ensure only columns present in both train and test are used for modeling)

# -------------------- Phase 2: Data Preprocessing & Feature Engineering --------------------

"""Phase 2: Data Preprocessing & Feature Engineering

    2.1.Combine Data (Optional but Recommended):
        Often, it's easier to preprocess train and test data together after initial split (or combine them, preprocess, then split).
        For this project, you can decide whether to process them separately or concatenate them first and then split again.
        For now, let's keep it simple: process train.csv for training, and then apply the same steps to test.csv for final prediction.
    2.2.Handle Missing Values: Decide on strategies:
        Numerical: Impute with mean, median, or 0.
        Categorical: Impute with mode, or a new category like 'None'.
        Drop columns with too many missing values.
    2.3.Handle Categorical Features:
        Apply One-Hot Encoding (for nominal categories) or Label Encoding (for ordinal categories, if applicable).
    2.4.Feature Engineering (Start Simple):
        Maybe create a simple feature like TotalSF = GrLivArea + 1stFlrSF + 2ndFlrSF (or similar).
    2.5.Feature Scaling: Prepare numerical features for models that require it (like Linear Regression) using StandardScaler.
        (Remember trees don't need it, but it's good practice for the overall pipeline).
    
"""

# 2.1. Decide whether to combine train and test for preprocessing.
# For simplicity, process train and test separately, but ensure the same transformations are applied to both.

# 2.2. Handle Missing Values

# List of columns with >15-20% missing values (from EDA)
high_missing_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"]

# Drop high-missing columns from both train and test
train_df = train_df.drop(columns=high_missing_cols)
test_df = test_df.drop(columns=high_missing_cols)

# Impute categorical columns with 'None'
for col in train_categorical_cols:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna("None")
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna("None")

# Impute numerical columns with median
for col in train_numerical_cols:
    if col in train_df.columns:
        median = train_df[col].median()
        train_df[col] = train_df[col].fillna(median)
    if col in test_df.columns:
        # Use train median to avoid data leakage
        test_df[col] = test_df[col].fillna(median)  # type: ignore

# 2.3. Convert Numeric-Categorical Features to string/object dtype
numeric_categorical = ["MSSubClass", "MoSold", "YrSold"]
for col in numeric_categorical:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype(str)
    if col in test_df.columns:
        test_df[col] = test_df[col].astype(str)

# 2.4. Feature Engineering: Create TotalSF (Total Square Footage)
for df in [train_df, test_df]:
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

# 2.5. Target Variable Transformation (log1p for right-skewed SalePrice)
train_df["SalePrice_log"] = np.log1p(train_df["SalePrice"])

# 2.6. One-Hot Encoding for categorical features
# Get updated categorical columns after type conversion and dropping
categorical_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
# Exclude 'SalePrice' if present
if "SalePrice" in categorical_cols:
    categorical_cols.remove("SalePrice")

# Apply get_dummies to both train and test, aligning columns
train_encoded = pd.get_dummies(train_df, columns=categorical_cols)
test_encoded = pd.get_dummies(test_df, columns=categorical_cols)
train_encoded, test_encoded = train_encoded.align(
    test_encoded, join="left", axis=1, fill_value=0
)

# 2.7. Feature Scaling for numerical features (excluding target)
scaler = StandardScaler()
numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
# Remove target columns from scaling
for target_col in ["SalePrice", "SalePrice_log"]:
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
# Scale train and test numerical features
train_encoded[numerical_cols] = scaler.fit_transform(train_encoded[numerical_cols])
test_encoded[numerical_cols] = scaler.transform(test_encoded[numerical_cols])

# 2.8. Final Data Ready for Modeling
# X = features, y = target (log-transformed)
X = train_encoded.drop(["SalePrice", "SalePrice_log"], axis=1)
y = train_encoded["SalePrice_log"]

# For test set, drop target columns if present
X_test = test_encoded.drop(["SalePrice", "SalePrice_log"], axis=1, errors="ignore")

# Print shapes for sanity check
print(f"Processed train shape: {X.shape}")
print(f"Processed test shape: {X_test.shape}")

# Ready for Phase 3: Model Selection, Training & Evaluation


# -------------------- Phase 3: Model Selection, Training & Evaluation (Enhanced) --------------------

"""Phase 3: Model Selection, Training & Evaluation

    3.1. Data Preparation
        Confirm all preprocessing steps from Phase 2 are complete (no missing values, aligned columns, correct dtypes).
        Split processed training data into features (X) and target (y).

    3.2. Train/Validation Split & Cross-Validation
        Split data into training and validation sets (e.g., 80/20 split).
        Set a random seed for reproducibility.
        Use k-fold cross-validation (e.g., cv=5) for robust model evaluation and hyperparameter tuning.

    3.3. Model Selection & Training
        Train a Linear Regression model as a baseline.
        Train a Decision Tree Regressor.
        Train a Random Forest Regressor.
        Use GridSearchCV (with cross-validation) to tune hyperparameters for tree-based models.
        
    3.4. Model Evaluation
        Evaluate models using:
            Mean Squared Error (MSE)
            Root Mean Squared Error (RMSE)
            R-squared (R²)
        Compare model performance on validation data.
        Plot residuals (predicted vs. actual) for error analysis.
        Investigate outliers and error patterns.

    3.5. Feature Importance & Selection
        Examine feature importances from tree-based models.
        Consider removing or engineering features based on importance and error analysis.
        Optionally, use feature selection techniques (e.g., SelectFromModel, Recursive Feature Elimination).

    3.6. Ensembling (Optional, for higher accuracy)
        Try simple ensembling (e.g., averaging predictions from multiple models).
        Optionally, try stacking or blending for advanced ensembling.

    3.7. Final Model Training
        Retrain the best model (or ensemble) on the full training data.
        Prepare the test set features (X_test) using the same preprocessing pipeline.

    3.8. Submission Preparation
        Predict on the test set.
        Inverse-transform the target (apply np.expm1 if log-transform was used).
        Prepare the submission file in the required Kaggle format (Id, SalePrice).
        Double-check for missing or misaligned columns in the submission.

    3.9. Documentation & Reproducibility
        Document all modeling choices and results.
        Save random seeds and model parameters for reproducibility.
        Summarize findings, best model, and next steps.
"""
# 3.1. Data Preparation
# Confirm all preprocessing steps from Phase 2 are complete (no missing values, aligned columns, correct dtypes).
# Split processed training data into features (X) and target (y).
X = train_encoded.drop(["SalePrice", "SalePrice_log"], axis=1)
y = train_encoded["SalePrice_log"]
# For test set, drop target columns if present
X_test = test_encoded.drop(["SalePrice", "SalePrice_log"], axis=1, errors="ignore")
# 3.2. Train/Validation Split & Cross-Validation
# Split data into training and validation sets (e.g., 80/20 split).
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
# 3.3. Model Selection & Training

# Train a Linear Regression model as a baseline.
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)

# Train a Decision Tree Regressor.
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Train a Random Forest Regressor.
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Use GridSearchCV (with cross-validation) to tune hyperparameters for tree-based models.
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
grid_search = GridSearchCV(
    rf_model, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=2
)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_

# 3.3b. Cross-validation for all models (for robust comparison)
print("\nCross-validation scores (RMSE, 5-fold):")
cv_lr = -cross_val_score(
    baseline_model, X, y, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
)
print(f"Linear Regression: Mean RMSE = {cv_lr.mean():.4f} (+/- {cv_lr.std():.4f})")
cv_dt = -cross_val_score(
    dt_model, X, y, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
)
print(f"Decision Tree:     Mean RMSE = {cv_dt.mean():.4f} (+/- {cv_dt.std():.4f})")
cv_rf = -cross_val_score(
    best_rf_model, X, y, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
)
print(f"Random Forest:     Mean RMSE = {cv_rf.mean():.4f} (+/- {cv_rf.std():.4f})")

# 3.4. Model Evaluation
# Evaluate models using:
mse_baseline = mean_squared_error(y_val, baseline_model.predict(X_val))
rmse_baseline = np.sqrt(mse_baseline)
r2_baseline = r2_score(y_val, baseline_model.predict(X_val))
print(
    f"\nBaseline Model: MSE={mse_baseline:.4f}, RMSE={rmse_baseline:.4f}, R²={r2_baseline:.4f}"
)
mse_dt = mean_squared_error(y_val, dt_model.predict(X_val))
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_val, dt_model.predict(X_val))
print(f"Decision Tree Model: MSE={mse_dt:.4f}, RMSE={rmse_dt:.4f}, R²={r2_dt:.4f}")
mse_rf = mean_squared_error(y_val, best_rf_model.predict(X_val))
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_val, best_rf_model.predict(X_val))
print(f"Random Forest Model: MSE={mse_rf:.4f}, RMSE={rmse_rf:.4f}, R²={r2_rf:.4f}")

# 3.4b. Deeper Error Analysis
residuals_rf = y_val - best_rf_model.predict(X_val)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_val, best_rf_model.predict(X_val), alpha=0.5, color="red")
plt.xlabel("Actual SalePrice (log)")
plt.ylabel("Predicted SalePrice (log)")
plt.title("Random Forest: Predicted vs Actual (log)")
plt.subplot(1, 2, 2)
plt.hist(residuals_rf, bins=30, color="orange", edgecolor="k")
plt.title("Random Forest: Residuals Histogram")
plt.xlabel("Residual (Actual - Predicted)")
plt.tight_layout()
plt.show()

# 3.5. Feature Importance & Selection
feature_importances = best_rf_model.feature_importances_
df_importance = pd.DataFrame({"Feature": X.columns, "Importance": feature_importances})
df_importance = df_importance.sort_values("Importance", ascending=False)
print("\nTop 10 Feature Importances (Random Forest):")
print(df_importance.head(10))

# 3.6. Ensembling (Optional, for higher accuracy)
# Simple averaging ensemble of best models
ensemble = VotingRegressor(
    estimators=[
        ("lr", baseline_model),
        ("dt", dt_model),
        ("rf", best_rf_model),
    ],
    n_jobs=-1,
)
ensemble.fit(X_train, y_train)
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble.predict(X_val)))
print(f"\nEnsemble Model (VotingRegressor): RMSE={ensemble_rmse:.4f}")

# 3.7. Final Model Training
# Retrain the best model (or ensemble) on the full training data.
final_model = best_rf_model  # or use 'ensemble' for ensembling
final_model.fit(X, y)

# 3.8. Submission Preparation
# Predict on the test set.
y_pred = final_model.predict(X_test)
# Inverse-transform the target (apply np.expm1 if log-transform was used).
y_pred = np.expm1(y_pred)
# Prepare the submission file in the required Kaggle format (Id, SalePrice).
submission = pd.DataFrame({"Id": test_raw["Id"], "SalePrice": y_pred})
submission.to_csv("Mini-Project_Predict_House_Prices/submission.csv", index=False)
print("\nSample submission:")
print(submission.head())

# 3.9. Documentation & Reproducibility
# Save random seeds and model parameters for reproducibility.
joblib.dump(final_model, "Mini-Project_Predict_House_Prices/final_model.joblib")
print("\nModel saved to 'final_model.joblib'.")

# Summarize findings, best model, and next steps.
print(
    "\nPhase 3 complete. Review RMSEs, feature importances, and consider further tuning or ensembling for higher accuracy."
)
