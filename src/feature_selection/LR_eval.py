###
# IMPORTS
###
import os
import pandas as pd
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, auc, precision_recall_curve, r2_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import torch
import seaborn as sns

###
# DEFINE CONSTANTS
###

# program parameters
cohort = "pMCIiAD" # pMCIiAD pHCiAD
eval = True
plot = True

###
# DATA PROCESSING
###

data = pd.read_csv("processed/master_data.csv").sort_values(by=["RID", "EXAMDATE_RANK"])

# find correct columns
begin_met = data.columns.get_loc("L_HISTIDINE")
end_met = data.columns.get_loc("BUDCA") + 1
begin_BA = data.columns.get_loc("CA")
begin_BA_ratio = data.columns.get_loc("CA_CDCA")
end_BA_ratio = data.columns.get_loc("GLCA_CDCA") + 1

# log 10 scale
data.iloc[:, begin_met:end_met] = np.log10(data.iloc[:, begin_met:end_met].replace(0, np.nan))
data.iloc[:, begin_BA_ratio:begin_BA_ratio] = np.log10(data.iloc[:, begin_BA_ratio:begin_BA_ratio].replace(0, np.nan))

# isolate iAD patients
data["DX_VALS"] = data["DX_VALS"].replace(3, 2)
rids = []
for rid, patient in data.groupby("RID"):
    dxs_og = patient["DX_VALS"].values
    dxs = np.where(dxs_og == 1, 2, dxs_og)
    if 2 in dxs:
        idx_2 = list(dxs).index(2)
        if 4 in dxs[idx_2:]:
            idx_4 = list(dxs[idx_2:]).index(4) + idx_2
            if 2 not in dxs[idx_4:] and 1 not in dxs[idx_2:idx_4]:
                rids.append(rid)

iAD = data[data["RID"].isin(rids)]

if cohort == "pHCiAD":
    # isoloate pHC patients
    pHC = data.groupby("RID").filter(lambda x: (x["DX_VALS"] == 1).all())
    # combine cohorts
    pHCiAD = pd.concat([pHC, iAD], ignore_index=True)
elif cohort == "pMCIiAD":
    # isoloate pHC patients
    pMCI = data.groupby("RID").filter(lambda x: x["DX_VALS"].isin([2, 3]).all())
    # combine cohorts - TODO change pHCiAD later
    pHCiAD = pd.concat([pMCI, iAD], ignore_index=True)
    pHCiAD.to_csv("processed/pMCIiAD.csv", index=False)
else:
    raise ValueError("Invalid cohort")

X = []
y = []
for rid, patient in pHCiAD.groupby("RID"):
    bl = patient[patient["VISCODE2"] == "bl"]

    # only checking bl
    if bl.empty:
        continue

    met_data = [float(x) for x in bl.iloc[0, begin_met:end_met].values.tolist()] + \
               [float(x) for x in bl.iloc[0, begin_BA_ratio:end_BA_ratio].values.tolist()] + \
               [float(bl.iloc[0][covariate]) for covariate in ['AGE', 'PTGENDER', 'BMI', 'fast', 'APOE_e2e4']]
    X.append(met_data)
    if 4 in patient["DX_VALS"].values:
        y.append(1)
    else:
        y.append(0)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
X = torch.tensor(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

###
# LR
###

logr = linear_model.LogisticRegression(max_iter=10000)
logr.fit(X_train,y_train)

pred_probs = logr.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, pred_probs)
print(auroc)

fpr, tpr, _ = roc_curve(y_test, pred_probs)
pred_labels = (pred_probs > 0.5).astype(int)

accuracy = accuracy_score(y_test, pred_labels)
roc_auc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(y_test, pred_labels)
aproc = auc(recall, precision)
r2 = r2_score(y_test, pred_labels)

print("\n--- PERFORMANCE ---")
print(f"{cohort} LR")
print(f"accuracy: {accuracy:.4f}")
print(f"roc: {roc_auc:.4f}")
print(f"aproc: {aproc:.4f}")
print(f"R^2: {r2:.4f}")
print("-------------------")

if plot:
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{cohort}")
    plt.legend(loc="lower right")

    # plt.savefig(f'/figures/LR_ROC_{cohort}_{auroc:.2f}.png')
    plt.show()

    cm = confusion_matrix(y_test, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
