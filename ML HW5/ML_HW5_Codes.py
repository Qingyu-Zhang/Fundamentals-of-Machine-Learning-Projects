# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 08:41:42 2025

@author: Qingyu Zhang
"""

# Homework 5 Revised Code (Minimal Change Version)

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

data = pd.read_csv("wines.csv")
X = data.values



#################################
# 1. PCA
#################################
print("Question 1:")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

eigenvalues = pca.explained_variance_
print("Eigenvalues:")
print(eigenvalues)

eigens_above_one = np.sum(eigenvalues > 1)
print(f"Number of Eigenvalues above 1: {eigens_above_one}")

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, c='green')
plt.title("PCA Projection On 2D")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(True)
plt.show()

explained_variance_2d = np.sum(pca.explained_variance_ratio_[:2])
print(f"Variance explained by the top 2 principal components: {explained_variance_2d:.4f}")

print()



#################################
# 2. t-SNE and KL divergence vs perplexity
#################################
print("Question 2:")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

perplexities = np.arange(5, 155, 10)
kl_divergences = []

for perp in perplexities:
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    tsne.fit_transform(X_scaled)
    kl_divergences.append(tsne.kl_divergence_)

plt.figure(figsize=(6, 4))
plt.plot(perplexities, kl_divergences, marker='o')
plt.title('KL Divergence vs Perplexity (t-SNE)')
plt.xlabel('Perplexity')
plt.ylabel('KL Divergence')
plt.grid(True)
plt.show()

tsne_20 = TSNE(n_components=2, perplexity=20, random_state=42)
X_embedded = tsne_20.fit_transform(X_scaled)

plt.figure(figsize=(6, 4))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.7, c='orange')
plt.title('t-SNE 2D Embedding (Perplexity=20)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.grid(True)
plt.show()

print()



#################################
# 3. MDS
#################################
print("Question 3:")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42)
X_mds = mds.fit_transform(X_scaled)

stress = mds.stress_
print(f"Calculated Stress of MDS Embedding: {stress:.4f}")

plt.figure(figsize=(8,6))
plt.scatter(X_mds[:, 0], X_mds[:, 1], alpha=0.7, c='red')
plt.title(f"MDS 2D Embedding (Stress={stress:.4f})")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.show()

print()



#################################
# 4. Silhouette + kMeans (on PCA 2D)
#################################
print("Question 4:")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

silhouette_scores = []
k_values = range(2, 15)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8,6))
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score vs Number of Clusters (Performed on 2D PCA)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

best_k = k_values[np.argmax(silhouette_scores)]
print(f"Best number of clusters (k) picked by Silhouette Score: {best_k}")

kmeans_final = KMeans(n_clusters=best_k, random_state=42)
final_labels = kmeans_final.fit_predict(X_pca)


plt.figure(figsize=(8,6))
for cluster in np.unique(final_labels):
    plt.scatter(X_pca[final_labels == cluster, 0], X_pca[final_labels == cluster, 1], label=f'Cluster {cluster}')
plt.title('KMeans Clustering on 2D PCA')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
plt.grid(True)
plt.show()

total_inertia = kmeans_final.inertia_
print(f"Total sum of Squred Distances to respective cluster centers (Inertia): {total_inertia:.4f}")

print()



#################################
# 5. DBSCAN Clustering (with K-distance Method on PCA 2D)
#################################
print("Question 5:")

# Step 1: Data Standardization and PCA to 2D
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 2: Determine epsilon using K-distance graph
from sklearn.neighbors import NearestNeighbors

k = 5  # initial min_samples guess
nearest_neighbors = NearestNeighbors(n_neighbors=k)
nearest_neighbors.fit(X_pca)
distances, indices = nearest_neighbors.kneighbors(X_pca)

# Sort distances to plot
sorted_distances = np.sort(distances[:, k-1])

plt.figure(figsize=(8,6))
plt.plot(sorted_distances)
plt.title(f'K-distance Graph (k={k})')
plt.xlabel('Sorted Points')
plt.ylabel(f'Distance to {k}th Nearest Neighbor')
plt.grid(True)
plt.show()

# Select epsilon as distance at 90th percentile
epsilon_index = int(0.9 * len(sorted_distances))
epsilon = sorted_distances[epsilon_index]
print(f"Chosen epsilon based on K-distance graph: {epsilon:.4f}")

# Step 3: Tune min_samples to maximize Silhouette Score
from sklearn.metrics import silhouette_score

min_samples_candidates = range(4, 11)
best_score = -1
best_min_samples = None
best_labels = None

for min_samples in min_samples_candidates:
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X_pca)
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    if n_clusters >= 2:
        mask = cluster_labels != -1  # Exclude noise for silhouette calculation
        try:
            score = silhouette_score(X_pca[mask], cluster_labels[mask])
            if score > best_score:
                best_score = score
                best_min_samples = min_samples
                best_labels = cluster_labels
        except:
            continue

print(f"Best min_samples: {best_min_samples}")
print(f"Best Silhouette Score: {best_score:.4f}")

# Step 4: Apply DBSCAN with selected parameters
dbscan_final = DBSCAN(eps=epsilon, min_samples=best_min_samples)
final_labels = dbscan_final.fit_predict(X_pca)

# Step 5: Plot clustering result
plt.figure(figsize=(8,6))
unique_labels = np.unique(final_labels)
for label in unique_labels:
    if label == -1:
        plt.scatter(X_pca[final_labels == label, 0], X_pca[final_labels == label, 1],
                    c='k', marker='x', label='Noise')
    else:
        plt.scatter(X_pca[final_labels == label, 0], X_pca[final_labels == label, 1],
                    label=f'Cluster {label}')
plt.title(f'DBSCAN Clustering (eps={epsilon:.2f}, min_samples={best_min_samples})')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
plt.grid(True)
plt.show()

# Step 6: Report clustering statistics
n_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)
n_noise = np.sum(final_labels == -1)
print(f"Estimated number of clusters (excluding noise): {n_clusters}")
print(f"Number of noise points: {n_noise}")

print()



