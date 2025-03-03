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
delta_t = 24

###
# PARSE DATA
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
# isolate AD cases
iAD = data[data["RID"].isin(rids)]

pMCI = data.groupby("RID").filter(lambda x: x["DX_VALS"].isin([2, 3]).all())
pMCIiAD = pd.concat([pMCI, iAD], ignore_index=True)
pMCIiAD.to_csv("processed/pMCIiAD.csv", index=False)

for rid, patient in pMCIiAD.groupby("RID"):
    dxs = patient["DX_VALS"].values
    viscodes = patient["VISCODE2"].values
    viscode_int = []
    incident = np.nan
    found_first = False

    for i in range(len(patient)):
        curr_int = int(viscodes[i].replace("bl","0").replace("m",""))
        viscode_int.append(curr_int)
        if dxs[i] == 4 and not found_first:
            incident = curr_int
            found_first = True

    if not np.isnan(incident):
        diff = [x - incident for x in viscode_int]
    else:
        diff = [np.nan for x in viscode_int]

    pMCIiAD.loc[pMCIiAD["RID"] == rid, "DIFF"] = diff

pMCIiAD.to_csv("processed/pMCIiAD.csv", index=False)

sixmoAD = pMCIiAD[pMCIiAD["DIFF"] == delta_t]
blMCI = pMCI[pMCI["VISCODE2"] == "bl"]

sixmoAD2 = sixmoAD.assign(y=1)
blMCI2 = blMCI.assign(y=0)
both = pd.concat([sixmoAD2, blMCI2], ignore_index=True)
both.to_csv("processed/blMCI-6dt.csv", index=False)

X_sixmoAD = pd.DataFrame(np.concatenate([sixmoAD.iloc[:, begin_met:end_met].values,
                            sixmoAD.iloc[:, begin_BA_ratio:end_BA_ratio].values,
                            sixmoAD[['AGE', 'PTGENDER', 'BMI', 'fast', 'APOE_e2e4']].values], axis=1))

X_blMCI = pd.DataFrame(np.concatenate([blMCI.iloc[:, begin_met:end_met].values,
                          blMCI.iloc[:, begin_BA_ratio:end_BA_ratio].values,
                          blMCI[['AGE', 'PTGENDER', 'BMI', 'fast', 'APOE_e2e4']].values], axis=1))
X_sixmoAD = X_sixmoAD.dropna()
X_blMCI = X_blMCI.dropna()
X = pd.concat([X_sixmoAD, X_blMCI], axis=0)
y = np.concatenate(([1] * len(X_sixmoAD), [0] * len(X_blMCI)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
print(X_sixmoAD.shape[0], "cases")
print(X_blMCI.shape[0], "controls")
print(f"{cohort} LR {delta_t}")
print(f"accuracy: {accuracy:.4f}")
print(f"roc: {roc_auc:.4f}")
print(f"auprc: {aproc:.4f}")
print("-------------------")