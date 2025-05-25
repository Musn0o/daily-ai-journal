import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  # Our SVM Classifier!
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# --- Synthetic Dataset (from previous exercises, for SVM) ---
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

"""Your SVM Challenge: Implement and Evaluate Support Vector Machines!"""

"""1.Train-Test Split:

    Split X_exercise and y_exercise into training and testing sets. Use test_size=0.2 and random_state=42.
"""
X_train, X_test, y_train, y_test = train_test_split(
    X_exercise, y_exercise, test_size=0.2, random_state=42
)

"""2.Feature Scaling (ESSENTIAL for SVM!):

    Instantiate a StandardScaler.
    Apply it to your training features (X_train) using fit_transform().
    Apply the same fitted scaler to your test features (X_test) using transform().
"""
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""3.Model Training (SVC):

    Instantiate an SVC model.
    Crucially, experiment with different kernel types:
        First, try kernel='linear'. This will build a linear decision boundary.
        Then, try kernel='rbf' (Radial Basis Function). This is a very powerful non-linear kernel, often a good default.
        (Optional Challenge: For rbf kernel,
        you can also experiment with the C parameter (e.g., C=0.1, C=1, C=10) and gamma
        (e.g., gamma='scale', gamma=0.1, gamma=1)).)
    Remember to set random_state=42 for reproducibility if you use a version of SVC that requires it 
    (though for SVC with deterministic kernels like linear or rbf, it's less critical, but good practice).
    Train the model using your scaled training features (X_train_scaled) and your training target (y_train)."""

# first try linear kernel
svm = SVC(kernel="linear", random_state=42, C=1, gamma="scale")
svm.fit(X_train_scaled, y_train)
# second try rbf kernel
svm_2nd = SVC(kernel="rbf", random_state=42, C=10, gamma="scale")
svm_2nd.fit(X_train_scaled, y_train)


"""4.Make Predictions:

    Use the trained model to predict class labels (0 or 1) on your X_test_scaled. Store these as y_pred."""

y_pred = svm.predict(X_test_scaled)
y_pred_2nd = svm_2nd.predict(X_test_scaled)
print("Predictions (y_pred):\n", y_pred)
print("Predictions (y_pred_2nd):\n", y_pred_2nd)
print("----------------------------------------------------\n")

"""5.Model Evaluation:

    Calculate and print the accuracy_score between y_test and y_pred.
    Print the full classification_report using y_test and y_pred.
    Print the confusion_matrix using y_test and y_pred.
"""
print("Accuracy Score (Linear Kernel):", accuracy_score(y_test, y_pred))
print("Classification Report (Linear Kernel):\n", classification_report(y_test, y_pred))
print("Confusion Matrix (Linear Kernel):\n", confusion_matrix(y_test, y_pred))
print("----------------------------------------------------\n")
print("Accuracy Score (RBF Kernel):", accuracy_score(y_test, y_pred_2nd))
print(
    "Classification Report (RBF Kernel):\n", classification_report(y_test, y_pred_2nd)
)
print("Confusion Matrix (RBF Kernel):\n", confusion_matrix(y_test, y_pred_2nd))

plt.scatter(
    X_exercise["exam_score"],
    X_exercise["hours_studied"],
    c=y_exercise,
    cmap="bwr",
    alpha=0.7,
)
plt.xlabel("Exam Score")
plt.ylabel("Hours Studied")
plt.title("Synthetic Data: Exam Score vs. Hours Studied")
plt.show()


# # Assume X_train_scaled, X_test_scaled, y_train, y_test are already defined as in your code

# C_values = [0.01, 0.1, 1, 10, 100, 1000]
# results = []

# for C in C_values:
#     svm = SVC(kernel="linear", C=C, random_state=42)
#     svm.fit(X_train_scaled, y_train)
#     y_pred = svm.predict(X_test_scaled)
#     acc = accuracy_score(y_test, y_pred)
#     results.append((C, acc))

# print("C vs. accuracy (linear kernel):")
# for C, acc in results:
#     print(f"C={C}: accuracy={acc:.3f}")

# best_C, best_acc = max(results, key=lambda x: x[1])
# print(f"\nBest C: {best_C} with accuracy: {best_acc:.3f}")

# # Optionally, print the classification report and confusion matrix for the best C
# svm_best = SVC(kernel="linear", C=best_C, random_state=42)
# svm_best.fit(X_train_scaled, y_train)
# y_pred_best = svm_best.predict(X_test_scaled)
# print("\nClassification report for best C (linear kernel):")
# print(classification_report(y_test, y_pred_best))
# print("Confusion matrix for best C (linear kernel):")
# print(confusion_matrix(y_test, y_pred_best))
