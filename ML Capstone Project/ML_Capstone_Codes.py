# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 08:33:01 2025

@author: Qingyu Zhang
"""

import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap.umap_ as umap


random.seed(19903322)
np.random.seed(19903322)

# ==== Step 1: Data Loading and Preparation ====
print('\n Step 1: Data Loading and Preparation')
df = pd.read_csv("musicData.csv")
df = df.dropna(subset=['instance_id'])  # Drop corrupted row
df['tempo'] = pd.to_numeric(df['tempo'], errors='coerce')
df['duration_ms'] = df['duration_ms'].replace(-1, np.nan)
df.drop(columns=['instance_id', 'artist_name', 'track_name', 'obtained_date'], inplace=True)
df.dropna(inplace=True)


# ==== Step 2: Feature Encoding ====
print('\n Step 2: Feature Encoding')
df['key'] = df['key'].astype('category').cat.codes
df = pd.get_dummies(df, columns=['mode'], drop_first=True)
genre_encoder = LabelEncoder()
df['genre_encoded'] = genre_encoder.fit_transform(df['music_genre'])



# ==== Step 3: Train/Test Split by Genre ====
print('\n Step 3: Train/Test Split by Genre')
# Sample 500 instances per genre for test set using groupby
test_set = df.groupby("genre_encoded", group_keys=False).apply(
    lambda x: x.sample(n=500, random_state=19903322)
)
train_set = df.drop(index=test_set.index)

# Separate features and labels
X_train = train_set.drop(columns=["music_genre", "genre_encoded"])
y_train = train_set["genre_encoded"]
X_test = test_set.drop(columns=["music_genre", "genre_encoded"])
y_test = test_set["genre_encoded"]





# ==== Step 4: Feature Scaling (excluding dummy variable) ====
print('\n Step 4: Feature Scaling (excluding dummy variable)')
# Identify dummy and numeric features
dummy_cols = ['mode_Minor']
numeric_cols = [col for col in X_train.columns if col not in dummy_cols]

# Scale only the numeric columns
scaler = StandardScaler()
X_train_numeric_scaled = scaler.fit_transform(X_train[numeric_cols])
X_test_numeric_scaled = scaler.transform(X_test[numeric_cols])

# Convert back to DataFrame for easier merging
X_train_scaled_df = pd.DataFrame(X_train_numeric_scaled, columns=numeric_cols, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_numeric_scaled, columns=numeric_cols, index=X_test.index)


X_train_scaled = pd.concat([X_train_scaled_df, X_train[dummy_cols]], axis=1)
X_test_scaled = pd.concat([X_test_scaled_df, X_test[dummy_cols]], axis=1)




# ==== Step 5: Random Forest + GridSearchCV ====
print('\n Step 5: Random Forest + GridSearchCV')
param_grid = {
    'n_estimators': [50, 100, 300],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=19903322)
grid = GridSearchCV(rf, param_grid, scoring='roc_auc_ovr', cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train_scaled, y_train)

best_rf = grid.best_estimator_
print("Best Parameters:", grid.best_params_)



# ==== Step 6: Feature Importance Plot ====
print('\n Step 6: Feature Importance Plot')
importances = best_rf.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df.sort_values(by='Importance', ascending=True, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title("Random Forest (Tuned) Feature Importances")
plt.tight_layout()
plt.show()



# ==== Step 7: Multi-Class AUC and ROC Curves ====
print('\n Step 7: Multi-Class AUC and ROC Curves')
# Binarize test labels for multi-class AUC and ROC
y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
y_score = best_rf.predict_proba(X_test_scaled)
genre_names = genre_encoder.inverse_transform(np.unique(y_train))

# Compute and plot ROC for each genre
plt.figure(figsize=(10, 7))
for i, genre in enumerate(genre_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    auc_score = roc_auc_score(y_test_bin[:, i], y_score[:, i])
    plt.plot(fpr, tpr, label=f"{genre} (AUC = {auc_score:.3f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest Multi-Class ROC Curve (After Hyperparameter Tuning)")
plt.legend(loc='lower right', fontsize='small')
plt.tight_layout()
plt.show()

macro_auc = roc_auc_score(y_test_bin, y_score, average='macro', multi_class='ovr')
print("Macro-Averaged AUC:", macro_auc)





# ==== Step 8: Direct t-SNE with PCA Initialization ====
print('\n Step 8 (Alternative): Dimensionality Reduction with t-SNE (init="pca")')

# Exclude dummy/categorical features before t-SNE
full_features = df.drop(columns=['music_genre', 'genre_encoded', 'mode_Minor'])
full_labels = df['music_genre']

# Standardize only continuous features
full_scaled = StandardScaler().fit_transform(full_features)

# Run t-SNE with PCA initialization
tsne = TSNE(n_components=2, init='pca', random_state=19903322, perplexity=30)
tsne_result = tsne.fit_transform(full_scaled)

# Create DataFrame for visualization
tsne_df = pd.DataFrame(tsne_result, columns=['TSNE1', 'TSNE2'])
tsne_df['Genre'] = full_labels.values

# Plot the t-SNE result
plt.figure(figsize=(10, 7))
sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue='Genre', palette='tab10', alpha=0.7)
plt.title("t-SNE Clustering of Songs by Genre (init='pca')")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()




# ==== Step 8 (Alternative): Dimensionality Reduction with UMAP and Genre Overlay ====
print('\n Step 8 (Alternative): Dimensionality Reduction with UMAP (Colored by Genre)')

# Prepare features (exclude categorical dummy and labels)
viz_features = df.drop(columns=['music_genre', 'genre_encoded', 'mode_Minor'])
viz_labels = df['music_genre']

# Standardize numeric features
viz_scaled = StandardScaler().fit_transform(viz_features)

# Apply UMAP to reduce to 2D
umap_model = umap.UMAP(n_components=2, random_state=19903322)
umap_result = umap_model.fit_transform(viz_scaled)

# Prepare dataframe for visualization
umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'])
umap_df['Genre'] = viz_labels.values

# Plot UMAP 2D projection with genre labels
plt.figure(figsize=(10, 7))
sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Genre', palette='tab10', alpha=0.7)
plt.title("UMAP Projection of Songs Colored by Genre")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()





# ==== Step 9: Unsupervised Clustering using UMAP + KMeans ====
print('\n Step 9: Clustering with UMAP and KMeans')

# Prepare data (exclude categorical)
unsup_features = df.drop(columns=['music_genre', 'genre_encoded', 'mode_Minor'])
unsup_scaled = StandardScaler().fit_transform(unsup_features)

# Step 9.1: Reduce dimensions with UMAP to 2D
reducer = umap.UMAP(n_components=2, random_state=19903322)
embedding = reducer.fit_transform(unsup_scaled)

# Step 9.2: Try multiple k values to find best number of clusters
silhouette_scores = []
k_values = list(range(2, 11))  # Try k=2 to k=10

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=19903322)
    cluster_labels = kmeans.fit_predict(embedding)
    score = silhouette_score(embedding, cluster_labels)
    silhouette_scores.append(score)
    print(f"k = {k}, Silhouette Score = {score:.4f}")

# Step 9.3: Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o')
plt.title("Silhouette Scores for Different K (UMAP + KMeans)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

# Step 9.4: Final KMeans with best k (highest silhouette score)
best_k = k_values[np.argmax(silhouette_scores)]
print(f"Best k selected based on silhouette score: {best_k}")

final_kmeans = KMeans(n_clusters=best_k, random_state=19903322)
final_labels = final_kmeans.fit_predict(embedding)

# Step 9.5: Plot UMAP with KMeans clusters
umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
umap_df['Cluster'] = final_labels.astype(str)

plt.figure(figsize=(10, 7))
sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Cluster', palette='tab10', alpha=0.8)
plt.title(f"UMAP + KMeans Clustering (k = {best_k})")
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Step 9.6: UMAP with KMeans using prior knowledge (k=10)
kmeans2 = KMeans(n_clusters=10, random_state=19903322)
labels2 = kmeans2.fit_predict(embedding)

umap_df2 = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
umap_df2['Cluster'] = labels2.astype(str)

plt.figure(figsize=(10, 7))
sns.scatterplot(data=umap_df2, x='UMAP1', y='UMAP2', hue='Cluster', palette='tab10', alpha=0.8)
plt.title("UMAP + KMeans Clustering (k = 10)")
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()








#%% Extra Credit

# List of numerical features including the label
numerical_features = [
    'popularity', 'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence',
    'tempo', 'duration_ms', 'key', 'genre_encoded'  # include label
]

# Set up the plot
plt.figure(figsize=(20, 25))

for i, feature in enumerate(numerical_features):
    plt.subplot(5, 3, i + 1)
    plt.hist(df[feature], bins=50, color='steelblue', edgecolor='black')
    plt.title(f'{feature} Distribution')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

