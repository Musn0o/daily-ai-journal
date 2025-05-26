import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import joblib
import datetime

# Set random seed for reproducibility
np.random.seed(42)

# -------------------- Phase 1: Data Acquisition & Initial Exploration (EDA) --------------------
train_df = pd.read_csv("Mini-Project_Predict_House_Prices/data/train.csv")
test_df = pd.read_csv("Mini-Project_Predict_House_Prices/data/test.csv")
train_raw = train_df.copy()
test_raw = test_df.copy()

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


def eda_report(df, target=None, name="DataFrame"):
    print(f"\n--- {name} HEAD ---")
    print(df.head())
    print(f"\n--- {name} INFO ---")
    print(df.info())
    print(f"\n--- {name} DESCRIBE ---")
    print(df.describe())
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(f"\nMissing values (sorted):\n{missing}")
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


train_categorical_cols, train_numerical_cols = eda_report(
    train_df, target="SalePrice", name="Train Data"
)
test_categorical_cols, test_numerical_cols = eda_report(test_df, name="Test Data")

# -------------------- Phase 2: Data Preprocessing & Feature Engineering --------------------
high_missing_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"]
train_df = train_df.drop(columns=high_missing_cols)
test_df = test_df.drop(columns=high_missing_cols)

for col in train_categorical_cols:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna("None")
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna("None")

for col in train_numerical_cols:
    if col in train_df.columns:
        median = train_df[col].median()
        train_df[col] = train_df[col].fillna(median)
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna(median)

numeric_categorical = ["MSSubClass", "MoSold", "YrSold"]
for col in numeric_categorical:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype(str)
    if col in test_df.columns:
        test_df[col] = test_df[col].astype(str)

for df in [train_df, test_df]:
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

train_df["SalePrice_log"] = np.log1p(train_df["SalePrice"])

categorical_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
if "SalePrice" in categorical_cols:
    categorical_cols.remove("SalePrice")

train_encoded = pd.get_dummies(train_df, columns=categorical_cols)
test_encoded = pd.get_dummies(test_df, columns=categorical_cols)
train_encoded, test_encoded = train_encoded.align(
    test_encoded, join="left", axis=1, fill_value=0
)

scaler = StandardScaler()
numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
for target_col in ["SalePrice", "SalePrice_log"]:
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
train_encoded[numerical_cols] = scaler.fit_transform(train_encoded[numerical_cols])
test_encoded[numerical_cols] = scaler.transform(test_encoded[numerical_cols])

X = train_encoded.drop(["SalePrice", "SalePrice_log"], axis=1)
y = train_encoded["SalePrice_log"]
X_test = test_encoded.drop(["SalePrice", "SalePrice_log"], axis=1, errors="ignore")

print(f"Processed train shape: {X.shape}")
print(f"Processed test shape: {X_test.shape}")


def add_polynomial_features(df, top_features, degree=2):
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    poly_features = poly.fit_transform(df[top_features])
    poly_feature_names = poly.get_feature_names_out(top_features)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
    df = df.drop(columns=top_features)
    df = pd.concat([df, poly_df], axis=1)
    return df


default_top_features = ["OverallQual", "GrLivArea", "TotalSF"]
fdefault_top_features = [feat for feat in default_top_features if feat in X.columns]
if fdefault_top_features:
    X = add_polynomial_features(X, fdefault_top_features, degree=2)
    X_test = add_polynomial_features(X_test, fdefault_top_features, degree=2)


# -------------------- Phase 3: Model Selection, Hyperparameter Tuning, and Evaluation --------------------
def print_cv_scores(model, X, y, name):
    scores = -cross_val_score(
        model, X, y, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    print(f"{name}: Mean RMSE = {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores.mean(), scores.std()


def evaluate_model(model, X_val, y_val, name):
    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, preds)
    print(f"{name}: MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    return mse, rmse, r2


def print_top_importances(model, X, name, topn=10):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        df_imp = pd.DataFrame({"Feature": X.columns, "Importance": importances})
        df_imp = df_imp.sort_values("Importance", ascending=False)
        print(f"\nTop {topn} Feature Importances ({name}):")
        print(df_imp.head(topn))
        return df_imp
    else:
        print(f"{name} does not support feature importances.")
        return None


# 3.1. Train/Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# 3.2. Model Definitions
baseline_model = LinearRegression()
dt_model = DecisionTreeRegressor(random_state=42)
rf_model = RandomForestRegressor(random_state=42)
gbr_model = GradientBoostingRegressor(random_state=42)

# 3.3. Hyperparameter Tuning (RandomizedSearchCV for speed)
rf_param_dist = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [None, 10, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2"],
}
rf_random_search = RandomizedSearchCV(
    rf_model,
    rf_param_dist,
    n_iter=20,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=2,
    random_state=42,
)
rf_random_search.fit(X_train, y_train)
best_rf_model = rf_random_search.best_estimator_

gbr_param_dist = {
    "n_estimators": [100, 200, 300, 400],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 4, 5, 6],
    "subsample": [0.8, 1.0],
    "min_samples_split": [2, 5, 10],
}
gbr_random_search = RandomizedSearchCV(
    gbr_model,
    gbr_param_dist,
    n_iter=15,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=2,
    random_state=42,
)
gbr_random_search.fit(X_train, y_train)
best_gbr_model = gbr_random_search.best_estimator_

# 3.4. Cross-validation and Validation for all models
models = [
    ("Linear Regression", baseline_model),
    ("Decision Tree", dt_model),
    ("Random Forest (Tuned)", best_rf_model),
    ("Gradient Boosting (Tuned)", best_gbr_model),
]
results = []

print("\nCross-validation scores (RMSE, 5-fold):")
for name, model in models:
    cv_mean, cv_std = print_cv_scores(model, X, y, name)
    model.fit(X_train, y_train)
    mse, rmse, r2 = evaluate_model(model, X_val, y_val, name)
    results.append(
        {
            "Model": name,
            "CV Mean RMSE": cv_mean,
            "CV Std RMSE": cv_std,
            "Validation RMSE": rmse,
            "Validation R²": r2,
            "Fitted Model": model,
        }
    )

# 3.5. Feature Importance (Random Forest & Gradient Boosting)
rf_importances = print_top_importances(best_rf_model, X, "Random Forest")
gbr_importances = print_top_importances(best_gbr_model, X, "Gradient Boosting")

# 3.6. Ensembling (VotingRegressor)
ensemble = VotingRegressor(
    estimators=[
        ("lr", baseline_model),
        ("rf", best_rf_model),
        ("gbr", best_gbr_model),
    ],
    n_jobs=-1,
)
ensemble.fit(X_train, y_train)
ensemble_preds = ensemble.predict(X_val)
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_preds))
ensemble_r2 = r2_score(y_val, ensemble_preds)
print(
    f"\nEnsemble Model (VotingRegressor): RMSE={ensemble_rmse:.4f}, R²={ensemble_r2:.4f}"
)
results.append(
    {
        "Model": "Ensemble (VotingRegressor)",
        "CV Mean RMSE": np.nan,
        "CV Std RMSE": np.nan,
        "Validation RMSE": ensemble_rmse,
        "Validation R²": ensemble_r2,
        "Fitted Model": ensemble,
    }
)

# 3.7. Results Summary Table
results_df = pd.DataFrame(results).drop(columns=["Fitted Model"])
print(f"\n{'=' * 60}\nPhase 3 Model Performance Summary\n{'=' * 60}")
print(results_df.to_string(index=False, float_format="%.4f"))

best_idx = results_df["Validation RMSE"].astype(float).idxmin()
best_model_name = results_df.loc[best_idx, "Model"]
print(f"\nBest model (lowest Validation RMSE): {best_model_name}")
print(f"{'=' * 60}\n")

# 3.8. Final Model Training & Submission Preparation
final_model = results[best_idx]["Fitted Model"]  # type: ignore
final_model.fit(X, y)
y_pred = final_model.predict(X_test)
if "SalePrice_log" in train_encoded.columns:
    y_pred = np.expm1(y_pred)

submission = pd.DataFrame({"Id": test_raw["Id"], "SalePrice": y_pred})
submission.to_csv("Mini-Project_Predict_House_Prices/submission.csv", index=False)
print("\nSample submission:")
print(submission.head())

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"Mini-Project_Predict_House_Prices/final_model_{timestamp}.joblib"
joblib.dump(final_model, model_path)
print(f"\nModel saved to '{model_path}'.")

print(
    "\nPhase 3 complete. Review the summary table above and consider further feature engineering or hyperparameter tuning for even better results."
)
