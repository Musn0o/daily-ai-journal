import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV

# No StandardScaler needed for Decision Trees, but you can keep it if you want consistency!
from sklearn.tree import DecisionTreeClassifier  # Our Decision Tree Classifier!
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Synthetic Dataset (from previous exercises, for Decision Tree) ---
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

"""Your Decision Tree Challenge: Implement and Evaluate a Decision Tree!"""

"""1.Train-Test Split:

    Split X_exercise and y_exercise into training and testing sets. Use test_size=0.2 and random_state=42.
"""
X_train, X_test, y_train, y_test = train_test_split(
    X_exercise, y_exercise, test_size=0.2, random_state=42
)

"""2.Model Training (DecisionTreeClassifier):

    Instantiate a DecisionTreeClassifier model. Remember to set random_state=42 for reproducibility.
    Crucially, experiment with max_depth:
        First, try with max_depth=None (the default). This allows the tree to grow as deep as it needs to make all leaves pure. Observe its performance. This often leads to overfitting on training data.
        Then, try limiting the depth, for example, max_depth=3 or max_depth=5. See how this affects your performance metrics and the complexity of the tree. This is a common way to "prune" a tree to reduce overfitting.
    Train the model using your training features (X_train) and your training target (y_train). (No scaling needed this time!).
"""
# param_grid = {
#     'max_depth': [2, 3, 4, 5, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
# grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
# grid_search.fit(X_train, y_train)
# print("Best parameters:", grid_search.best_params_)
# best_model = grid_search.best_estimator_
# y_pred_best = best_model.predict(X_test)
# print("Accuracy with best params:", accuracy_score(y_test, y_pred_best))


# First, try with max_depth=None
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)
# Limiting the depth, for example, max_depth=3 or max_depth=5
dtc_2 = DecisionTreeClassifier(max_depth=3, random_state=42)
dtc_2.fit(X_train, y_train)  # Train the model again with the new max_depth setting
# After testing Best parameters: are 'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2
dtc_best = DecisionTreeClassifier(
    max_depth=4, random_state=42, min_samples_leaf=1, min_samples_split=2
)
dtc_best.fit(X_train, y_train)  # Train the model again with the new max_depth setting

"""3.Make Predictions:

    Use the trained model to predict class labels (0 or 1) on your X_test. Store these as y_pred.
"""
# Now, let's make predictions
# First model with max_depth=None
y_pred = dtc.predict(X_test)
# Second model with max_depth=3
y_pred_2 = dtc_2.predict(X_test)
# Third model with max_depth=4
y_pred_best = dtc_best.predict(X_test)

"""4.Model Evaluation:

    Calculate and print the accuracy_score between y_test and y_pred.
    Print the full classification_report using y_test and y_pred.
    Print the confusion_matrix using y_test and y_pred.
"""
accuracy_score_1 = accuracy_score(y_test, y_pred)
print(f"Accuracy with max_depth=None: {accuracy_score_1}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("\n----------------------------------------------------\n")
accuracy_score_2 = accuracy_score(y_test, y_pred_2)
print(f"Accuracy with max_depth=3: {accuracy_score_2}")
print(classification_report(y_test, y_pred_2))
print(confusion_matrix(y_test, y_pred_2))
print("\n----------------------------------------------------\n")
accuracy_score_3 = accuracy_score(y_test, y_pred_best)
print(f"Accuracy with max_depth=4: {accuracy_score_3}")
print(classification_report(y_test, y_pred_best))
print(confusion_matrix(y_test, y_pred_best))  # type: ignore

"""Bonus Challenge (Highly Recommended for Decision Trees!): Visualize the Tree"""

# Plot the first model with max_depth=None
plt.figure(figsize=(10, 8))
plot_tree(
    dtc,
    filled=True,
    feature_names=X_exercise.columns,  # type: ignore
    class_names=["Passed", "Failed"],
)
plt.show()
# Plot the second model with max_depth=3
plt.figure(figsize=(10, 8))
plot_tree(
    dtc_2,
    filled=True,
    feature_names=X_exercise.columns,  # type: ignore
    class_names=["Passed", "Failed"],
)
plt.show()
# Plot the third model with max_depth=4
plt.figure(figsize=(10, 8))
plot_tree(
    dtc_best,
    filled=True,
    feature_names=X_exercise.columns,  # type: ignore
    class_names=["Passed", "Failed"],
)
plt.show()
