import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  # Our new model!
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Synthetic Dataset (from previous exercise, for KNN) ---
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

"""Your KNN Challenge: Implement and Evaluate K-Nearest Neighbors!"""
"""
    1.Train-Test Split:
        Split X_exercise and y_exercise into training and testing sets. Use test_size=0.2 and random_state=42.
        """
X_train, X_test, y_train, y_test = train_test_split(
    X_exercise, y_exercise, test_size=0.2, random_state=42
)

""" 2.Feature Scaling (CRUCIAL for KNN!):
        Instantiate a StandardScaler.
        Apply it to your training features (X_train) using fit_transform().
        Apply the same fitted scaler to your test features (X_test) using transform().
    """
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""3.Model Training (KNeighborsClassifier):
    Instantiate a KNeighborsClassifier model.
    Experiment with n_neighbors (the k value). Start with n_neighbors=5, then try n_neighbors=3, and observe the changes in performance metrics.
    Train the model using your scaled training features (X_train_scaled) and your training target (y_train).
"""
knn_model = KNeighborsClassifier(n_neighbors=15)  # Start with n_neighbors=5
knn_model.fit(X_train_scaled, y_train)


"""4.Make Predictions:
         Use the trained model to predict class labels (0 or 1) on your X_test_scaled. Store these as y_pred."""

y_pred = knn_model.predict(X_test_scaled)
print("Predictions (y_pred):\n", y_pred)
print("----------------------------------------------------\n")

"""5.Model Evaluation:
        Calculate and print the accuracy_score between y_test and y_pred.
        Print the full classification_report using y_test and y_pred.
        Print the confusion_matrix using y_test and y_pred.        
"""
# Calculate and print the accuracy_score between y_test and y_pred.
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.3f}")
# Print the full classification_report using y_test and y_pred.
print(classification_report(y_test, y_pred))
# Print the confusion_matrix using y_test and y_pred.
print(confusion_matrix(y_test, y_pred))


# Try different values for n_neighbors
# results = []
# for k in range(1, 16):
#     knn_model = KNeighborsClassifier(n_neighbors=k)
#     knn_model.fit(X_train_scaled, y_train)
#     """4.Make Predictions:
#         Use the trained model to predict class labels (0 or 1) on your X_test_scaled. Store these as y_pred."""
#     y_pred = knn_model.predict(X_test_scaled)
#     """5.Model Evaluation:
#         Calculate and print the accuracy_score between y_test and y_pred.
#         Print the full classification_report using y_test and y_pred.
#         Print the confusion_matrix using y_test and y_pred.
# """
#     acc = accuracy_score(y_test, y_pred)
#     results.append((k, acc))
# # Print results
# print("n_neighbors vs. accuracy:")
# for k, acc in results:
#     print(f"k={k}: accuracy={acc:.3f}")

# # Find the best k
# best_k, best_acc = max(results, key=lambda x: x[1])
# print(f"\nBest n_neighbors: {best_k} with accuracy: {best_acc:.3f}")
# # Optionally, print the classification report and confusion matrix for the best k
# knn_best = KNeighborsClassifier(n_neighbors=best_k)
# knn_best.fit(X_train_scaled, y_train)
# y_pred_best = knn_best.predict(X_test_scaled)
# print("\nClassification report for best k:")
# print(classification_report(y_test, y_pred_best))
# print("Confusion matrix for best k:")
# print(confusion_matrix(y_test, y_pred_best))
