###
# IMPORTS
###

import numpy as np
import pandas as pd
import torch
import sys, os
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score, roc_auc_score, auc, precision_recall_curve, r2_score, roc_curve, confusion_matrix

###
# DEFINE CONSTANTS
###

num_mets = 111
cohort = "pMCIiAD" # pHCiAD, pMCIiAD
imputed = True
delta_t = [-6,0]

###
# GET DATA
###

pMCIiAD = pd.read_csv("processed/pMCIiAD.csv")

# find correct columns
begin_met = pMCIiAD.columns.get_loc("L_HISTIDINE")
end_met = pMCIiAD.columns.get_loc("BUDCA") + 1
begin_BA = pMCIiAD.columns.get_loc("CA")
begin_BA_ratio = pMCIiAD.columns.get_loc("CA_CDCA")
end_BA_ratio = pMCIiAD.columns.get_loc("GLCA_CDCA") + 1

valid_rids_iAD = []
valid_rids_pMCI = []
for rid, patient in pMCIiAD.groupby("RID"):
    diffs = patient["DIFF"].values
    if all(dt in diffs for dt in delta_t):
        valid_rids_iAD.append(rid)
    elif np.any(np.isnan(diffs)) and len(diffs) > (len(delta_t) - 1):
        valid_rids_pMCI.append(rid)

deltaT_iAD = pMCIiAD[pMCIiAD["RID"].isin(valid_rids_iAD)]
deltaT_iAD = deltaT_iAD[deltaT_iAD["DIFF"].isin(delta_t)]
deltaT_iAD = deltaT_iAD[deltaT_iAD["DIFF"].isin(delta_t)].drop_duplicates(subset=["RID", "DIFF"], keep="first")

deltaT_iAD.to_csv("processed/deltaT_iAD.csv", index=False)
deltaT_pMCI = pMCIiAD[pMCIiAD["RID"].isin(valid_rids_pMCI)]
# print(len(valid_rids_pMCI), len(valid_rids_iAD))
# print(deltaT_pMCI.shape, deltaT_iAD.shape)

deltaT_pMCIiAD = pd.concat([deltaT_iAD, deltaT_pMCI])
# print(deltaT_pMCIiAD.shape)

###
# GET DIFFERENCES IN METABOLITES
###

X = []
y = []
for rid, patient in deltaT_pMCIiAD.groupby("RID"):
    met_data = np.concatenate([patient.iloc[:len(delta_t), begin_met:end_met].fillna(0).values,
                               patient.iloc[:len(delta_t), begin_BA_ratio:end_BA_ratio].fillna(0).values], axis=1)
    met_diff = met_data[-1] - met_data[0]
    curr_x = patient.iloc[0][["AGE", "BMI", "fast", "PTGENDER", "chol", "hdl", "trig"]].values.flatten().tolist() + met_diff.tolist()

    if not np.any(np.isnan(curr_x)):
        X.append(curr_x)
        if 4 in patient["DX_VALS"].values:
            y.append(1)
        else:
            y.append(0)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

logr = linear_model.LogisticRegression(max_iter=5000)
logr.fit(X_train,y_train)

pred_probs = logr.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, pred_probs)
# print(auroc)

fpr, tpr, _ = roc_curve(y_test, pred_probs)
pred_labels = (pred_probs > 0.5).astype(int)

accuracy = accuracy_score(y_test, pred_labels)
roc_auc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(y_test, pred_labels)
aproc = auc(recall, precision)
r2 = r2_score(y_test, pred_labels)

print("\n--- PERFORMANCE ---")
print(f"{cohort} LR {delta_t}")
print(f"accuracy: {accuracy:.4f}")
print(f"roc: {roc_auc:.4f}")
print(f"auprc: {aproc:.4f}")
print("-------------------")