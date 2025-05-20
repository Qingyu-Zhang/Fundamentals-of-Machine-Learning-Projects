# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 02:50:58 2025

@author: Qingyu Zhang
"""

# Re-import required packages due to kernel reset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay

# Reload dataset
df = pd.read_csv("diabetes.csv")

# Separate features and target
X = df.drop(columns="Diabetes")
y = df["Diabetes"]

# Train-test split (stratified sampling based on values in y, dealing with potential imbalanced classes problem)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)



#%% Question 1
print('Question 1:')


# Standardize full feature set for baseline model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train baseline logistic regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)
y_pred_proba = logreg.predict_proba(X_test_scaled)[:, 1]
baseline_auc = roc_auc_score(y_test, y_pred_proba)

#Draw the ROC for baesline model
RocCurveDisplay.from_predictions(
    y_test, 
    y_pred_proba, 
    name=f"Logistic Regression (AUC = {baseline_auc:.4f})",
    color="darkorange"
)

plt.plot([0, 1], [0, 1], linestyle="--", color="navy", label="Random guess")
plt.title("ROC Curve - Logistic Regression")
plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# Feature ablation using pandas
feature_aucs = {}
for col in X.columns:
    X_train_ablate = X_train.drop(columns=col)
    X_test_ablate = X_test.drop(columns=col)

    scaler_ablate = StandardScaler()
    X_train_scaled_ablate = scaler_ablate.fit_transform(X_train_ablate)
    X_test_scaled_ablate = scaler_ablate.transform(X_test_ablate)

    logreg_ablate = LogisticRegression(max_iter=1000)
    logreg_ablate.fit(X_train_scaled_ablate, y_train)
    y_pred_ablate = logreg_ablate.predict_proba(X_test_scaled_ablate)[:, 1]
    auc_ablate = roc_auc_score(y_test, y_pred_ablate)

    feature_aucs[col] = baseline_auc - auc_ablate

# Sort and return top 5 features by AUC drop
sorted_ablation = sorted(feature_aucs.items(), key=lambda x: x[1], reverse=True)
print('Baseline AUC (Logistic Regression):',baseline_auc)

print("\nTop 5 predictors by AUC drop:")
for i, (feature, drop) in enumerate(sorted_ablation[:5], 1):
    print(f"{i}. {feature}: AUC drop = {drop:.4f}")



print()




#%% Question2

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.pipeline import make_pipeline

print('Question 2:')

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# # Standardize features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)



# # 定义参数网格
# param_grid = {
#     "C": [0.01, 0.1, 1, 10],
#     "kernel": ["linear", "rbf"],
#     "max_iter": [2000]
# }

# # 网格搜索 + 交叉验证
# svm_grid = GridSearchCV(
#     estimator=SVC(probability=True, random_state=42),
#     param_grid=param_grid,
#     cv=5,
#     scoring="roc_auc",
#     n_jobs=-1  # 本地可设为 -1 使用所有核心
# )

# # 训练
# svm_grid.fit(X_train_scaled, y_train)

# # 拿到最佳模型
# svm_best = svm_grid.best_estimator_
# y_pred_proba_svm = svm_best.predict_proba(X_test_scaled)[:, 1]
# baseline_auc_svm = roc_auc_score(y_test, y_pred_proba_svm)

# # 画 ROC 曲线
# RocCurveDisplay.from_predictions(
#     y_test, y_pred_proba_svm,
#     name=f"SVM ({svm_grid.best_params_['kernel']}, C={svm_grid.best_params_['C']}, AUC={baseline_auc_svm:.4f})",
#     color="green"
# )
# plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
# plt.title("ROC Curve - SVM")
# plt.grid(True)
# plt.legend(loc="lower right")
# plt.tight_layout()
# plt.show()









# # 从 GridSearchCV 中获取最佳参数并加上 max_iter
# best_params = svm_grid.best_params_.copy()
# best_params["max_iter"] = 2000  # 确保 ablation 过程不会卡死



# # 记录每个特征删掉后的 AUC drop
# feature_aucs = {}

# # 对每个特征进行消融分析
# for col in tqdm(X.columns, desc="Ablation on SVM"):
#     # 删除当前特征
#     X_train_ablate = X_train.drop(columns=col)
#     X_test_ablate = X_test.drop(columns=col)

#     # 重新标准化
#     scaler_ablate = StandardScaler()
#     X_train_scaled_ablate = scaler_ablate.fit_transform(X_train_ablate)
#     X_test_scaled_ablate = scaler_ablate.transform(X_test_ablate)

#     # 重新用最佳参数训练 SVM
#     svm_ablate = SVC(**best_params, probability=True, random_state=42)
#     svm_ablate.fit(X_train_scaled_ablate, y_train)
#     y_pred_ablate = svm_ablate.predict_proba(X_test_scaled_ablate)[:, 1]

#     # 计算 ablated AUC 并记录 AUC drop
#     auc_ablate = roc_auc_score(y_test, y_pred_ablate)
#     feature_aucs[col] = baseline_auc_svm - auc_ablate

# # 排序输出最重要的前 5 个特征
# sorted_ablation_svm = sorted(feature_aucs.items(), key=lambda x: x[1], reverse=True)

# print(f"\nBaseline AUC (SVM): {baseline_auc_svm:.4f}")
# print("Top 5 predictors based on AUC drop when removed:")
# for i, (feature, drop) in enumerate(sorted_ablation_svm[:5], 1):
#     print(f"{i}. {feature} - AUC drop: {drop:.4f}")



# GridSearchCV 调参部分不变
param_grid_lsvc = {
    "linearsvc__C": [0.01, 0.1, 1, 10, 20, 30, 40, 50, 60, 70, 100, 500, 1000]
}

pipeline_lsvc = make_pipeline(
    StandardScaler(),
    LinearSVC(max_iter=2000, dual=False, random_state=42)
)

grid_lsvc = GridSearchCV(
    estimator=pipeline_lsvc,
    param_grid=param_grid_lsvc,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid_lsvc.fit(X_train, y_train)

# 拿到最佳模型并获取决策分数
lsvc_best = grid_lsvc.best_estimator_
y_scores_lsvc = lsvc_best.decision_function(X_test)
baseline_auc_lsvc = roc_auc_score(y_test, y_scores_lsvc)

# ✅ 用 RocCurveDisplay 来画图（从分数）
RocCurveDisplay.from_predictions(
    y_test,
    y_scores_lsvc,
    name=f"LinearSVC (C={grid_lsvc.best_params_['linearsvc__C']}, AUC = {baseline_auc_lsvc:.4f})",
    color="teal"
)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.title("ROC Curve - LinearSVC")
plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 输出结果
print(f"\nBaseline AUC (LinearSVC): {baseline_auc_lsvc:.4f}")
print("Best parameter from GridSearch:")
print(f"- C: {grid_lsvc.best_params_['linearsvc__C']}")


    
    

print()


best_C_lsvc = grid_lsvc.best_params_['linearsvc__C']

# Feature ablation loop
feature_aucs_lsvc = {}

for col in tqdm(X.columns, desc="Ablation on LinearSVC"):
    X_train_ablate = X_train.drop(columns=col)
    X_test_ablate = X_test.drop(columns=col)

    ablate_pipeline = make_pipeline(
        StandardScaler(),
        LinearSVC(C=best_C_lsvc, max_iter=2000, dual=False, random_state=42)
    )
    ablate_pipeline.fit(X_train_ablate, y_train)
    auc_ablate = roc_auc_score(y_test, ablate_pipeline.decision_function(X_test_ablate))

    feature_aucs_lsvc[col] = baseline_auc_lsvc - auc_ablate

# Sort and display top 5
sorted_ablation_lsvc = sorted(feature_aucs_lsvc.items(), key=lambda x: x[1], reverse=True)

print(f"\nBaseline AUC (LinearSVC): {baseline_auc_lsvc:.4f}")
print("Top 5 predictors based on AUC drop when removed:")
for i, (feature, drop) in enumerate(sorted_ablation_lsvc[:5], 1):
    print(f"{i}. {feature} - AUC drop: {drop:.4f}")
    
    
print()




#%% Question3
from sklearn.tree import DecisionTreeClassifier

print('Question 3:')

# 1. 训练单棵决策树模型
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_proba_tree = tree_clf.predict_proba(X_test)[:, 1]
baseline_auc_tree = roc_auc_score(y_test, y_pred_proba_tree)

# 2. 绘制 ROC 曲线
RocCurveDisplay.from_predictions(
    y_test, y_pred_proba_tree,
    name=f"Decision Tree (AUC = {baseline_auc_tree:.4f})",
    color="blue"
)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.title("ROC Curve - Decision Tree")
plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 3. 特征消融分析（不需要标准化）
feature_aucs_tree = {}
for col in tqdm(X.columns, desc="Ablation on Decision Tree"):
    X_train_ablate = X_train.drop(columns=col)
    X_test_ablate = X_test.drop(columns=col)

    tree_ablate = DecisionTreeClassifier(random_state=42)
    tree_ablate.fit(X_train_ablate, y_train)
    y_pred_ablate = tree_ablate.predict_proba(X_test_ablate)[:, 1]
    auc_ablate = roc_auc_score(y_test, y_pred_ablate)

    feature_aucs_tree[col] = baseline_auc_tree - auc_ablate

# 4. 输出最重要特征
sorted_ablation_tree = sorted(feature_aucs_tree.items(), key=lambda x: x[1], reverse=True)

print(f"\nBaseline AUC (Decision Tree): {baseline_auc_tree:.4f}")
print("Top 5 predictors based on AUC drop when removed:")
for i, (feature, drop) in enumerate(sorted_ablation_tree[:5], 1):
    print(f"{i}. {feature} - AUC drop: {drop:.4f}")
    


print()




#%% Question4
from sklearn.ensemble import RandomForestClassifier
print('Question 4:')
# Parameter grid
param_grid_rf = {
    "n_estimators": [100, 200],
    "max_features": ["sqrt", "log2"],
    "max_samples": [0.2, 0.5, 0.8]
}

# GridSearchCV on Random Forest
rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(
        bootstrap=True,
        criterion="gini",
        random_state=42,
        n_jobs=-1
    ),
    param_grid=param_grid_rf,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

# Fit the model
rf_grid.fit(X_train, y_train)

# Predict and calculate AUC
rf_best = rf_grid.best_estimator_
y_pred_proba_rf = rf_best.predict_proba(X_test)[:, 1]
baseline_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

# Plot ROC
RocCurveDisplay.from_predictions(
    y_test, y_pred_proba_rf,
    name=f"Random Forest (AUC = {baseline_auc_rf:.4f})",
    color="purple"
)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.title("ROC Curve - Random Forest")
plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Output best parameters and AUC
print(f"\nBaseline AUC (Random Forest): {baseline_auc_rf:.4f}")
print("Best parameters from GridSearch:")
for k, v in rf_grid.best_params_.items():
    print(f"- {k}: {v}")

print()






rf_best_params = rf_grid.best_params_.copy()
rf_best_params["bootstrap"] = True
rf_best_params["criterion"] = "gini"
rf_best_params["random_state"] = 42
rf_best_params["n_jobs"] = -1


# Feature ablation
feature_aucs_rf = {}

for col in tqdm(X.columns, desc="Ablation on Random Forest"):
    X_train_ablate = X_train.drop(columns=col)
    X_test_ablate = X_test.drop(columns=col)

    rf_ablate = RandomForestClassifier(**rf_best_params)
    rf_ablate.fit(X_train_ablate, y_train)
    y_pred_ablate = rf_ablate.predict_proba(X_test_ablate)[:, 1]
    auc_ablate = roc_auc_score(y_test, y_pred_ablate)

    feature_aucs_rf[col] = baseline_auc_rf - auc_ablate

# 输出前 5 个最重要的 predictor
sorted_ablation_rf = sorted(feature_aucs_rf.items(), key=lambda x: x[1], reverse=True)

print(f"\nBaseline AUC (Random Forest): {baseline_auc_rf:.4f}")
print("Top 5 predictors based on AUC drop when removed:")
for i, (feature, drop) in enumerate(sorted_ablation_rf[:5], 1):
    print(f"{i}. {feature} - AUC drop: {drop:.4f}")


print()





#%% Question 5
from sklearn.ensemble import AdaBoostClassifier
print('Question 5:')

# 3. 设置调参范围
param_grid_ada = {
    "n_estimators": [200, 300],
    "learning_rate": [1.0, 1.5]
}

# 4. GridSearchCV 寻找最佳 AdaBoost 模型
ada_grid = GridSearchCV(
    estimator=AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        random_state=42
    ),
    param_grid=param_grid_ada,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

ada_grid.fit(X_train, y_train)

# 5. 使用最佳模型预测
ada_best = ada_grid.best_estimator_
y_pred_proba_ada = ada_best.predict_proba(X_test)[:, 1]
baseline_auc_ada = roc_auc_score(y_test, y_pred_proba_ada)

# 6. 画 ROC 曲线
RocCurveDisplay.from_predictions(
    y_test, y_pred_proba_ada,
    name=f"AdaBoost (AUC = {baseline_auc_ada:.4f})",
    color="darkred"
)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.title("ROC Curve - AdaBoost")
plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 7. 打印最优参数和 AUC（统一风格）
print(f"\nBaseline AUC (AdaBoost): {baseline_auc_ada:.4f}")
print("Best parameters from GridSearch:")
for k, v in ada_grid.best_params_.items():
    print(f"- {k}: {v}")
    

print()






ada_best_params = ada_grid.best_params_.copy()
ada_best_params["base_estimator"] = DecisionTreeClassifier(max_depth=1)
ada_best_params["random_state"] = 42


# 特征消融分析
feature_aucs_ada = {}

for col in tqdm(X.columns, desc="Ablation on AdaBoost"):
    X_train_ablate = X_train.drop(columns=col)
    X_test_ablate = X_test.drop(columns=col)

    ada_ablate = AdaBoostClassifier(**ada_best_params)
    ada_ablate.fit(X_train_ablate, y_train)
    y_pred_ablate = ada_ablate.predict_proba(X_test_ablate)[:, 1]
    auc_ablate = roc_auc_score(y_test, y_pred_ablate)

    feature_aucs_ada[col] = baseline_auc_ada - auc_ablate

# 排序输出 top 5
sorted_ablation_ada = sorted(feature_aucs_ada.items(), key=lambda x: x[1], reverse=True)

print(f"\nBaseline AUC (AdaBoost): {baseline_auc_ada:.4f}")
print("Top 5 predictors based on AUC drop when removed:")
for i, (feature, drop) in enumerate(sorted_ablation_ada[:5], 1):
    print(f"{i}. {feature} - AUC drop: {drop:.4f}")




print()





#%% Extra Credit (a)
from sklearn.metrics import roc_curve
print('Extra Credit:')


# Compute FPR, TPR manually for each model
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_pred_proba)
fpr_lsvc, tpr_lsvc, _ = roc_curve(y_test, y_scores_lsvc)
fpr_tree, tpr_tree, _ = roc_curve(y_test, y_pred_proba_tree)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
fpr_ada, tpr_ada, _ = roc_curve(y_test, y_pred_proba_ada)

# Compute AUCs again (optional, for label)
auc_logreg = roc_auc_score(y_test, y_pred_proba)
auc_lsvc = roc_auc_score(y_test, y_scores_lsvc)
auc_tree = roc_auc_score(y_test, y_pred_proba_tree)
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
auc_ada = roc_auc_score(y_test, y_pred_proba_ada)

# Plot all curves in one plot
plt.figure(figsize=(10, 8))

plt.plot(fpr_logreg, tpr_logreg, label=f"Logistic Regression (AUC = {auc_logreg:.4f})")
plt.plot(fpr_lsvc, tpr_lsvc, label=f"LinearSVC (AUC = {auc_lsvc:.4f})")
plt.plot(fpr_tree, tpr_tree, label=f"Decision Tree (AUC = {auc_tree:.4f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.4f})")
plt.plot(fpr_ada, tpr_ada, label=f"AdaBoost (AUC = {auc_ada:.4f})")

# Add baseline and formatting
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison Across All Models (One Plot)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()





#%% Extra Credit (b)
import seaborn as sns
from scipy.stats import mannwhitneyu

# Compare distributions of MentalHealth and PhysicalHealth by Diabetes status
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

sns.boxplot(x="Diabetes", y="MentalHealth", data=df, ax=axs[0])
axs[0].set_title("Mental Health Days (Past 30) vs Diabetes")
axs[0].set_xlabel("Diabetes")
axs[0].set_ylabel("MentalHealth (0-30 days)")

sns.boxplot(x="Diabetes", y="PhysicalHealth", data=df, ax=axs[1])
axs[1].set_title("Physical Health Days (Past 30) vs Diabetes")
axs[1].set_xlabel("Diabetes")
axs[1].set_ylabel("PhysicalHealth (0-30 days)")

plt.tight_layout()
plt.show()


# Split data based on Diabetes status
mental_health_diab = df[df["Diabetes"] == 1]["MentalHealth"].dropna()
mental_health_nondi = df[df["Diabetes"] == 0]["MentalHealth"].dropna()

physical_health_diab = df[df["Diabetes"] == 1]["PhysicalHealth"].dropna()
physical_health_nondi = df[df["Diabetes"] == 0]["PhysicalHealth"].dropna()

# Mann-Whitney U test (non-parametric, since data is skewed count data)
mental_u, mental_p = mannwhitneyu(mental_health_diab, mental_health_nondi, alternative="two-sided")
physical_u, physical_p = mannwhitneyu(physical_health_diab, physical_health_nondi, alternative="two-sided")

print(f'mental health p-value: {mental_p}; physical health p-value: {physical_p}')
