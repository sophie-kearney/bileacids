###
# IMPORTS
###

import pandas as pd
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os, shap
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score, roc_auc_score, auc, precision_recall_curve, r2_score, roc_curve, confusion_matrix

pd.set_option('display.max_columns', None)

###
# CONSTANTS
###

cohort = "pMCIiAD" # pHCiAD, pMCIiAD
trained_model = "seed32_MaskedGRU_0.8267"

###
# LOAD DATA
###

data = pd.read_csv(f"processed/{cohort}/{trained_model}_predictions.csv")
data = data[['AGE', 'PTGENDER', 'APOE_e2e4', 'ADRiskScore', 'AD', "LongCovRiskScore"]]
data.dropna(inplace=True)

X = data[["AGE","PTGENDER","APOE_e2e4","ADRiskScore", "LongCovRiskScore"]]
y = data["AD"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logr = linear_model.LogisticRegression()
logr.fit(X_train,y_train)

pred_probs = logr.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, pred_probs)

fpr, tpr, _ = roc_curve(y_test, pred_probs)
pred_labels = (pred_probs > 0.5).astype(int)

accuracy = accuracy_score(y_test, pred_labels)
roc_auc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(y_test, pred_labels)
aproc = auc(recall, precision)
r2 = r2_score(y_test, pred_labels)

print("\n--- PERFORMANCE ---")
print(f"accuracy: {accuracy:.4f}")
print(f"roc: {roc_auc:.4f}")
print(f"auprc: {aproc:.4f}")
print("-------------------")

plt.plot(fpr, tpr, label=f"AUROC = {auroc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"LR AUROC using RNN Embeddings and Covariates")
plt.legend(loc="lower right")

plt.show()

cm = confusion_matrix(y_test, pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': logr.coef_[0]
})
feature_importance['Importance'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

print("\n--- FEATURE IMPORTANCE ---")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()

explainer = shap.Explainer(logr, X_train)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test, feature_names=X.columns)