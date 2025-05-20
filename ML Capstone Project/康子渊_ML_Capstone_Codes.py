# %%
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Step 1: Data Preparation
print("\n Step 1: Data Preparation")
# Set random seed based on NYU N-number
random.seed(12428625)
np.random.seed(12428625)

# Load the dataset
df = pd.read_csv("musicData.csv")

# Remove invalid duration entries
# duration_ms = -1
df = df[df["duration_ms"] > 0].copy()

# Convert 'tempo' to float (it may be string in some rows)
df["tempo"] = pd.to_numeric(df["tempo"], errors="coerce")

# Convert 'key' to numeric codes
df["key"] = df["key"].astype("category").cat.codes

# One-hot encode 'mode' (keep only one dummy column)
df = pd.get_dummies(df, columns=["mode"], drop_first=True)

# Encode 'music_genre' as label
genre_encoder = LabelEncoder()
df["genre_encoded"] = genre_encoder.fit_transform(df["music_genre"])

# Drop unused columns not relevant for modeling
df.drop(columns=[
    "instance_id", "artist_name", "track_name", "obtained_date", "music_genre"
], inplace=True)

# Drop rows with missing values in modeling-related columns only
important_features = [
    "popularity", "acousticness", "danceability", "duration_ms", "energy",
    "instrumentalness", "key", "liveness", "loudness", "speechiness",
    "tempo", "valence", "mode_Minor"
]
df.dropna(subset=important_features, inplace=True)

# Show label encoding map (optional)
print("Genre label encoding:")
for i, label in enumerate(genre_encoder.classes_):
    print(f"{i}: {label}")

# Final confirmation
print("Final cleaned data shape:", df.shape)
print("Modeling features:", list(df.columns.drop('genre_encoded')))


# %%
# Step 2: Train/Test Split
print("\n Step 2: Train/Test Split")
# Correct split - do NOT reset_index!
test_set = df.groupby("genre_encoded", group_keys=False).apply(
    lambda x: x.sample(n=500, random_state=12428625)
)

# All remaining samples go to training set
train_set = df.drop(index=test_set.index)

# Then safely split features and labels
X_train = train_set.drop(columns=["genre_encoded"])
y_train = train_set["genre_encoded"]
X_test = test_set.drop(columns=["genre_encoded"])
y_test = test_set["genre_encoded"]

# Check shapes
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# %%
# Step 3: Model Building and Evaluation
print("\n Step 3: Model Building and Evaluation")
# Define hyperparameter search space
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 3],
    'max_features': ['sqrt', 'log2']
}

# Setup GridSearchCV
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=12428625),
    param_grid=param_grid,
    cv=3,                                # 3-fold cross-validation
    scoring='roc_auc_ovr',
    n_jobs=-1,                           # Use all CPU cores
    verbose=2
)

# Run Grid Search
grid_search.fit(X_train, y_train)

# Get the best model
best_rf_model = grid_search.best_estimator_

# Predict on test set
y_pred = best_rf_model.predict(X_test)
y_proba = best_rf_model.predict_proba(X_test)

# Binarize labels for multi-class AUC
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

# Compute macro AUC
auc_score = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")

# Classification report
print("=== Best Hyperparameters ===")
print(grid_search.best_params_)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=genre_encoder.classes_))

print(f"\nFinal Macro-Averaged AUC after tuning: {auc_score:.3f}")

# %%
# Step 4: Feature Importance
print("\n Step 4: Feature Importance")
# Extract feature importances from best_rf_model
importances = best_rf_model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis", hue="Feature")
plt.title("Random Forest (Tuned) Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Print feature ranking
print("Top Features Used by Tuned Random Forest:")
print(importance_df.to_string(index=False))

# %%
# Step 5: AUC and ROC Curves
print("\n Step 5: AUC and ROC Curves")
# Binarize y_test for multi-class ROC & AUC
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

# Calculate per-class AUC
n_classes = y_test_bin.shape[1]
per_class_auc = {}
for i in range(n_classes):
    auc = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
    per_class_auc[genre_encoder.classes_[i]] = auc

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=genre_encoder.classes_,
            yticklabels=genre_encoder.classes_)
plt.title("Confusion Matrix (Best RF after Tuning)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# ROC curve plot per genre
plt.figure(figsize=(10, 7))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    plt.plot(fpr, tpr, label=f"{genre_encoder.classes_[i]} (AUC = {per_class_auc[genre_encoder.classes_[i]]:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("Random Forest Multi-Class ROC Curve (After Hyperparameter Tuning)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print all per-class AUC scores
print("\nPer-Class AUC Scores:")
for genre, auc in per_class_auc.items():
    print(f"{genre:<12}: {auc:.3f}")

# %%
# Step 6: Clustering in Low-Dimensional Space
print("\n Step 6: Clustering in Low-Dimensional Space")
# Run UMAP on test data
umap_model = umap.UMAP(
    n_neighbors=30,
    min_dist=0.05,
    n_components=2,
    random_state=12428625
)
X_test_umap = umap_model.fit_transform(X_test)

# Try multiple k values for KMeans
range_k = range(2, 11)
best_score = -1
best_k = None
best_labels = None

for k in range_k:
    kmeans = KMeans(n_clusters=k, random_state=12428625, n_init='auto')
    labels = kmeans.fit_predict(X_test_umap)
    score = silhouette_score(X_test_umap, labels)
    print(f"Silhouette Score for k={k}: {score:.3f}")
    
    if score > best_score:
        best_score = score
        best_k = k
        best_labels = labels

# Store result
umap_kmeans_df = pd.DataFrame({
    "UMAP1": X_test_umap[:, 0],
    "UMAP2": X_test_umap[:, 1],
    "Cluster": best_labels
})

# Step 4: Plot final result
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x="UMAP1", y="UMAP2",
    hue="Cluster",
    palette="tab10",
    data=umap_kmeans_df,
    s=60,
    alpha=0.8
)
plt.title(f"UMAP + KMeans Clustering (k={best_k}, Silhouette Score={best_score:.3f})")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
# Step 7: Extra Credit - Non-Trivial Observation Plot
print("\n Step 7: Extra Credit - Speechiness Comparison between Hip-Hop and Rap")

# Add genre labels back to the test set
test_set_with_labels = test_set.copy()
test_set_with_labels["genre_name"] = genre_encoder.inverse_transform(y_test)

# Select only Hip-Hop and Rap samples
hiphop_rap_df = test_set_with_labels[test_set_with_labels["genre_name"].isin(["Hip-Hop", "Rap"])]

# Plot the Speechiness distribution comparison
plt.figure(figsize=(10,6))
sns.histplot(
    data=hiphop_rap_df,
    x="speechiness",
    hue="genre_name",
    kde=True,
    palette="Set1",
    bins=30,
    alpha=0.7
)
plt.title("Speechiness Distribution: Hip-Hop vs Rap")
plt.xlabel("Speechiness")
plt.ylabel("Density")
plt.legend(title="Genre")
plt.tight_layout()
plt.show()


