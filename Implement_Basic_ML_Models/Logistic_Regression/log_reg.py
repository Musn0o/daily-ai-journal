import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler  # You'll need this!
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)  # Crucial for classification evaluation!

"""Logistic Regression Exercise: Predicting Exam Pass/Fail

You have a dataset containing exam_score and hours_studied for students.
Your task is to build a Logistic Regression model to predict whether a student passed_exam (1) or failed_exam (0).

Here's the synthetic dataset you'll use:"""
# --- Synthetic Dataset for Logistic Regression Exercise ---
np.random.seed(42)  # For reproducibility

# Features: 'exam_score', 'hours_studied'
X_logistic_exercise = pd.DataFrame(
    {
        "exam_score": np.random.normal(70, 10, 200),  # Exam scores around 70
        "hours_studied": np.random.normal(10, 3, 200),  # Hours studied around 10
    }
)

# Target: 'passed_exam' (1 if passed, 0 if failed)
# This target is designed to be relatively balanced
linear_combination = (
    0.8 * X_logistic_exercise["exam_score"] + 1.2 * X_logistic_exercise["hours_studied"]
) - 60
prob_passed = 1 / (1 + np.exp(-linear_combination))  # Sigmoid to get probabilities
y_logistic_exercise = (prob_passed > 0.5 + (np.random.rand(200) - 0.5) * 0.2).astype(
    int
)  # Add noise and use 0.5 threshold

print("Features (X_logistic_exercise head):\n", X_logistic_exercise.head())
print("\nTarget (y_logistic_exercise head):\n", y_logistic_exercise.head())  # type: ignore
print(
    f"\nTarget distribution (0s vs 1s): {np.bincount(y_logistic_exercise)}"
)  # Check balance!

"""Your Challenge: Build, Train, Predict, and Evaluate Logistic Regression!

    1.Train-Test Split:
        Split X_logistic_exercise and y_logistic_exercise into training and testing sets. Use test_size=0.2 and random_state=42."""

X_train, X_test, y_train, y_test = train_test_split(
    X_logistic_exercise, y_logistic_exercise, test_size=0.2, random_state=42
)


"""2.Feature Scaling:

    Instantiate a StandardScaler.
    Apply it to your training features (X_train) using fit_transform().
    Apply the same fitted scaler to your test features (X_test) using transform()."""

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""3.Model Training:

    Instantiate a LogisticRegression model. (It's good practice to set random_state=42 here too for consistency,
    as some internal solvers use randomness).
    Train the model using your scaled training features (X_train_scaled) and your training target (y_train)."""

log_reg_model = LogisticRegression(random_state=42)
log_reg_model.fit(X_train_scaled, y_train)

"""4.Make Predictions:

    Use the trained model to predict class labels (0 or 1) on your X_test_scaled. Store these as y_pred_class.
    Use the trained model to predict probabilities for each class on your X_test_scaled.
    Store these as y_pred_proba. (Remember, predict_proba returns an array where each row is [probability_of_class_0, probability_of_class_1])."""

# Use your chosen threshold (e.g., best_thresh = 0.75)
best_thresh = 0.6000000000000001

# y_pred_class = log_reg_model.predict(X_test_scaled)
# Get predicted probabilities for class 1
y_pred_proba = log_reg_model.predict_proba(X_test_scaled)[:, 1]
# Apply the threshold to get final class predictions
y_pred_custom = (y_pred_proba > best_thresh).astype(int)

# # Try different thresholds to reduce FP
# thresholds = np.arange(0.5, 0.91, 0.05)
# results = []

# for thresh in thresholds:
#     y_pred_custom = (y_pred_proba > thresh).astype(int)
#     cm = confusion_matrix(y_test, y_pred_custom)
#     tn, fp, fn, tp = cm.ravel()
#     acc = accuracy_score(y_test, y_pred_custom)
#     results.append(
#         {"threshold": thresh, "accuracy": acc, "FP": fp, "TP": tp, "TN": tn, "FN": fn}
#     )

# results_df = pd.DataFrame(results)
# print("Threshold tuning results:\n", results_df)

# # Find the threshold(s) with FP == 0 and highest accuracy
# best = results_df[results_df["FP"] == 0]
# if not best.empty:
#     print("\nThreshold(s) with 0 FP and their accuracy:\n", best)
#     # Optionally, show the confusion matrix and report for the best threshold
#     best_thresh = best.iloc[0]["threshold"]
#     y_pred_best = (y_pred_proba > best_thresh).astype(int)
#     print(
#         f"\nClassification report for threshold {best_thresh}:\n",
#         classification_report(y_test, y_pred_best),
#     )
#     print(
#         f"\nConfusion matrix for threshold {best_thresh}:\n",
#         confusion_matrix(y_test, y_pred_best),
#     )
# else:
#     print("\nNo threshold in the tested range achieved 0 FP.")

"""5.Model Evaluation:

    Calculate and print the accuracy_score between y_test and y_pred_class.
    Print the full classification_report using y_test and y_pred_class.
    Pay close attention to the precision, recall, and f1-score for both class 0 and class 1.
    Print the confusion_matrix using y_test and y_pred_class.
    This will visually show you True Positives, True Negatives, False Positives, and False Negatives."""

accuracy = accuracy_score(y_test, y_pred_custom)
print(f"\nAccuracy: {accuracy}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_custom))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_custom))
