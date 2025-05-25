import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# No StandardScaler needed for Random Forest!
from sklearn.ensemble import RandomForestClassifier  # Our Random Forest Classifier!
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
)  # You mastered this, let's keep it in mind!

# --- Synthetic Dataset (from previous exercises, for Random Forest) ---
np.random.seed(42)  # For reproducibility

X_exercise = pd.DataFrame(
    {
        "exam_score": np.random.normal(70, 10, 200),
        "hours_studied": np.random.normal(10, 3, 200),
    }
)
linear_combination = (
    0.8 * X_exercise["exam_score"] + 1.2 * X_exercise["hours_studied"]
) - 60
prob_passed = 1 / (1 + np.exp(-linear_combination))
y_exercise = (prob_passed > 0.5 + (np.random.rand(200) - 0.5) * 0.2).astype(int)

print("Features (X_exercise head):\n", X_exercise.head())
print("\nTarget (y_exercise head):\n", y_exercise.head())  # type: ignore
print(f"\nTarget distribution (0s vs 1s): {np.bincount(y_exercise)}")
print("\n----------------------------------------------------\n")

"""Your Random Forest Challenge: Implement and Evaluate a Random Forest!"""

"""1.Train-Test Split:

    Split X_exercise and y_exercise into training and testing sets. Use test_size=0.2 and random_state=42.
"""
X_train, X_test, y_train, y_test = train_test_split(
    X_exercise, y_exercise, test_size=0.2, random_state=42
)

"""2.Model Training (RandomForestClassifier):

    Instantiate a RandomForestClassifier model. Remember to set random_state=42 for reproducibility.
    Experiment with n_estimators (the number of trees):
        First, start with a reasonable default, like n_estimators=100.
        Then, try a higher value, for example, n_estimators=200 or n_estimators=500, and see if it changes the performance.
    (Optional Advanced Challenge): Since you've mastered GridSearchCV, feel free to use it to tune not only n_estimators but also max_depth (for individual trees) or max_features if you're feeling ambitious!
    Train the model using your training features (X_train) and your training target (y_train). (No scaling needed!).
"""
# param_grid = {
#     "n_estimators": [100, 200, 500],
#     "max_depth": [None, 10, 20, 30],
#     "max_features": [None, "sqrt", "log2"],
# }
# rf = RandomForestClassifier(random_state=42)
# grid_search = GridSearchCV(rf, param_grid, cv=5)
# grid_search.fit(X_train, y_train)
# best_rf = grid_search.best_estimator_
# print("Best Random Forest Parameters:", best_rf.get_params())
# print("Best Random Forest Score:", best_rf.score(X_test, y_test))

"""Best Random Forest Parameters: {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}
Best Random Forest Score: 0.975"""
rf = RandomForestClassifier(
    random_state=42, n_estimators=100, max_depth=None, max_features="sqrt"
)
rf.fit(X_train, y_train)

"""3.Make Predictions:

    Use the trained model to predict class labels (0 or 1) on your X_test. Store these as y_pred.
"""
y_pred = rf.predict(X_test)

"""4.Model Evaluation:

    Calculate and print the accuracy_score between y_test and y_pred.
    Print the full classification_report using y_test and y_pred.
    Print the confusion_matrix using y_test and y_pred.
"""
print("Accuracy of Random Forest:", accuracy_score(y_test, y_pred))
print("Confusion Matrix of Random Forest:\n", confusion_matrix(y_test, y_pred))
print(
    "Classification Report of Random Forest:\n", classification_report(y_test, y_pred)
)
