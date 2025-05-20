# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 03:02:29 2025

@author: Qingyu Zhang
"""

# Homework 4 - Diabetes Neural Network Tasks

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error, roc_curve
import matplotlib.pyplot as plt

# Add this for GPU toggle
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", DEVICE)

# Load dataset
df = pd.read_csv("diabetes.csv")

# Feature/target for classification (tasks 1-3): predict diabetes status (column 0) from all other columns
X_class = df.iloc[:, 1:].values
y_class = df.iloc[:, 0].values.reshape(-1, 1)

# Feature/target for regression (tasks 4-5): predict BMI (column 3) from all other columns except BMI
X_reg = df.drop(columns=["BMI"]).values if "BMI" in df.columns else np.delete(df.values, 3, axis=1)
y_reg = df.iloc[:, 3].values.reshape(-1, 1)  # BMI

# Normalize features
scaler_class = StandardScaler()
X_class_scaled = scaler_class.fit_transform(X_class)

scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

# Convert to tensors
X_class_tensor = torch.tensor(X_class_scaled, dtype=torch.float32).to(DEVICE)
y_class_tensor = torch.tensor(y_class, dtype=torch.float32).to(DEVICE)

X_reg_tensor = torch.tensor(X_reg_scaled, dtype=torch.float32).to(DEVICE)
y_reg_tensor = torch.tensor(y_reg, dtype=torch.float32).to(DEVICE)

# Train/test split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class_tensor, y_class_tensor, test_size=0.2, random_state=42
)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg_tensor, y_reg_tensor, test_size=0.2, random_state=42
)

# Dataloaders
train_dl_class = DataLoader(TensorDataset(X_train_c, y_train_c), batch_size=64, shuffle=True)
test_dl_class = DataLoader(TensorDataset(X_test_c, y_test_c), batch_size=64)

train_dl_reg = DataLoader(TensorDataset(X_train_r, y_train_r), batch_size=64, shuffle=True)
test_dl_reg = DataLoader(TensorDataset(X_test_r, y_test_r), batch_size=64)

###############################################
# ======= MODEL DEFINITIONS & UTILITIES =======
###############################################

class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.fc(x)

class FeedforwardNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation=None):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(last_dim, h))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class CNNNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * input_dim, 1)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

def train_binary(model, train_loader, test_loader, epochs=20, lr=0.001):
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    all_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds = torch.sigmoid(model(xb))
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_preds)

def train_regression(model, train_loader, test_loader, epochs=20, lr=0.001):
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb)
            preds.extend(pred.cpu().numpy())
            truths.extend(yb.cpu().numpy())
    return np.array(preds), np.array(truths)

def plot_roc(y_true, y_score, title):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    return auc


def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

###############################################
# ======= TASK-BY-TASK MODEL EXECUTION =======
###############################################

# TASK 1: Perceptron
model1 = Perceptron(input_dim=X_class.shape[1])
preds1 = train_binary(model1, train_dl_class, test_dl_class)
auc1 = plot_roc(y_test_c.cpu().numpy(), preds1, "Perceptron_ROC")
print("[Task 1] AUC (Perceptron):", auc1)

# TASK 2: FFNs with different activations (1 hidden layer and 2 hidden layers)
auc_table = []

for depth, layers in [(1, [32]), (2, [64, 32])]:
    for act in [None, 'sigmoid', 'relu']:
        model = FeedforwardNet(input_dim=X_class.shape[1], hidden_layers=layers, activation=act)
        preds = train_binary(model, train_dl_class, test_dl_class)
        auc = plot_roc(y_test_c.cpu().numpy(), preds, f"MLP-{depth}-{act if act else 'None'}_ROC")
        print(f"[Task 2] AUC (MLP-{depth}, act={act}):", auc)
        auc_table.append({
            "Model": f"MLP-{depth}-{act if act else 'None'}",
            "Hidden Size": str(tuple(layers)),
            "Test AUC": round(auc, 3)
        })

# Add perceptron baseline
auc_table.append({"Model": "Perceptron (Q1)", "Hidden Size": "N/A", "Test AUC": round(auc1, 3)})

# Display table
auc_df = pd.DataFrame(auc_table)
print("Summary Table (Task 2):")
print(auc_df.to_string(index=False))

# TASK 3: Deep FFN vs CNN
deep_ffn = FeedforwardNet(input_dim=X_class.shape[1], hidden_layers=[64, 32], activation='relu')
deep_preds = train_binary(deep_ffn, train_dl_class, test_dl_class)
deep_auc = plot_roc(y_test_c.cpu().numpy(), deep_preds, "Deep_FFN_ROC")
print("[Task 3] AUC (Deep FFN):", deep_auc)

cnn_model = CNNNet(input_dim=X_class.shape[1])
cnn_preds = train_binary(cnn_model, train_dl_class, test_dl_class)
cnn_auc = plot_roc(y_test_c.cpu().numpy(), cnn_preds, "CNN_ROC")
print("[Task 3] AUC (CNN):", cnn_auc)

# TASK 4: Regression - FFN with activations
for act in [None, 'sigmoid', 'relu']:
    model4 = FeedforwardNet(input_dim=X_reg.shape[1], hidden_layers=[32], activation=act)
    preds4, truths4 = train_regression(model4, train_dl_reg, test_dl_reg)
    rmse4 = compute_rmse(truths4, preds4)
    print(f"[Task 4] RMSE (act={act}):", rmse4)

# TASK 5: Best regression model
model5 = FeedforwardNet(input_dim=X_reg.shape[1], hidden_layers=[64, 32, 16], activation='relu')
preds5, truths5 = train_regression(model5, train_dl_reg, test_dl_reg)
rmse5 = compute_rmse(truths5, preds5)
print("[Task 5] Best RMSE:", rmse5)
























#%% Extra Credits

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Load dataset for extra credit
extra_df = pd.read_csv("diabetes.csv")


X_extra = extra_df.drop(columns=["Diabetes"])
y_extra = extra_df["Diabetes"]

# Normalize data
scaler = StandardScaler()
X_scaled_extra = scaler.fit_transform(X_extra)

# Extra Credit (a): Mutual Information for Feature Importance
mi_scores = mutual_info_classif(X_scaled_extra, y_extra, random_state=42)
mi_df = pd.DataFrame({
    "Feature": X_extra.columns,
    "Mutual Information": mi_scores
}).sort_values(by="Mutual Information", ascending=False)

print("\nExtra Credit (a) - Feature Importance via Mutual Information:")
print(mi_df.to_string(index=False))


