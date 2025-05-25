import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans  # Our K-Means model!
from sklearn.metrics import silhouette_score  # For evaluating clusters


# --- Synthetic Dataset (from previous exercises, for K-Means) ---
np.random.seed(42)  # For reproducibility

X_exercise = pd.DataFrame(
    {
        "exam_score": np.random.normal(70, 10, 200),
        "hours_studied": np.random.normal(10, 3, 200),
    }
)
# Keep y_exercise for later comparison, though K-Means doesn't use it for fitting
linear_combination = (
    0.8 * X_exercise["exam_score"] + 1.2 * X_exercise["hours_studied"]
) - 60
prob_passed = 1 / (1 + np.exp(-linear_combination))
y_exercise = (prob_passed > 0.5 + (np.random.rand(200) - 0.5) * 0.2).astype(
    int
)  # Our 'true' labels for comparison

print("Features (X_exercise head):\n", X_exercise.head())
print(
    f"\nNote: For K-Means, we fit on the full data. We have 'y_exercise' for comparison later: {np.bincount(y_exercise)}"
)
print("\n----------------------------------------------------\n")
"""Your K-Means Challenge: Implement and Evaluate K-Means Clustering!"""

"""1.Feature Scaling (CRUCIAL!):

    Instantiate a StandardScaler.
    Apply it to your entire X_exercise dataset using fit_transform(). Store the result in a new variable, e.g., X_scaled.
"""
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_exercise)


"""2.Determine Optimal k (Elbow Method):

    Create an empty list, e.g., inertia_values = [].
    Loop through a range of k values (e.g., from 1 to 10).
    Inside the loop:
        Instantiate KMeans(n_clusters=k, random_state=42, n_init=10) (set n_init=10 to suppress a future warning and ensure robust centroid initialization).
        Fit the KMeans model on your X_scaled data.
        Append the model.inertia_ to your inertia_values list.
    Plot the Elbow Method:
        Plot k (the range you looped through) on the x-axis against inertia_values on the y-axis.
        Look for the "elbow" point where the decrease in inertia starts to slow down significantly.
        This often suggests the optimal k. (Given our synthetic data's underlying structure, you might expect a clear elbow around k=2!)
"""
inertia_values = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia_values, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means")
plt.show()
optimal_k = 3  # Based on the Elbow Method, we expect k=2 or 3 to be optimal
print(f"Optimal k: {optimal_k}")


"""3.Model Training (K-Means with chosen k):

    Based on your Elbow Method plot, choose the optimal k (likely 2 for our dataset).
    Instantiate a new KMeans model with your chosen n_clusters=k, random_state=42, and n_init=10.
    Fit this model on your X_scaled data.
"""
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(X_scaled)


"""4.Get Cluster Labels and Centroids:

    Retrieve the cluster assignments for each data point using model.labels_. Store these in cluster_labels.
    Retrieve the coordinates of the cluster centroids using model.cluster_centers_. Store these in cluster_centers.
"""
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

score = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette Score (k={optimal_k}): {score:.3f}")

"""5.Visualize the Clusters:

    Create a scatter plot of your X_scaled data.
    Color the data points based on their cluster_labels.
    On the same plot, you can also add markers for the cluster_centers_ (e.g., using plt.scatter(cluster_centers[:, 0],
    cluster_centers[:, 1], marker='X', s=200, color='red', label='Centroids')).
    Add a title and legend.
    Bonus Comparison: Create a separate scatter plot of the original X_exercise data (unscaled, or scaled X_scaled)
    but color the points based on the original y_exercise labels.
    Compare this "true" grouping to the clusters found by K-Means.
    How well did K-Means "rediscover" the underlying pass/fail classes?
"""
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap="viridis")
plt.scatter(
    cluster_centers[:, 0],
    cluster_centers[:, 1],
    marker="X",
    s=200,
    color="red",
    label="Centroids",
)
plt.title("K-Means Clustering")
plt.xlabel("Exam Score")
plt.ylabel("Hours Studied")
plt.legend()
plt.show()
print("Cluster Centers:\n", cluster_centers)
print("Cluster Labels:\n", cluster_labels)
