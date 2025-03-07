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
delta_t = [-12, -6, 0]

###
# LOAD DATA
###

pMCIiAD = pd.read_csv("processed/pMCIiAD.csv")

# find correct columns
begin_met = pMCIiAD.columns.get_loc("L_HISTIDINE")
end_met = pMCIiAD.columns.get_loc("BUDCA") + 1
begin_BA = pMCIiAD.columns.get_loc("CA")
begin_BA_ratio = pMCIiAD.columns.get_loc("CA_CDCA")
end_BA_ratio = pMCIiAD.columns.get_loc("GLCA_CDCA") + 1
longitudinal_cov_columns = ["fast", "BMI", "trig", "chol","hdl" ]

###
# PARSE deltaT
###

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

deltaT_pMCI = pMCIiAD[pMCIiAD["RID"].isin(valid_rids_pMCI)]
print(len(valid_rids_iAD), len(valid_rids_pMCI))
# print(len(deltaT_iAD), deltaT_pMCI["RID"].nunique())
# print(deltaT_iAD.shape, deltaT_pMCI.shape)

###
# PROCESS DATA
###

X = []
y = []
rids = []
static_covariates = []
longitudinal_covariates = []

# get AD patients
for rid, patient in deltaT_iAD.groupby("RID"):
    met_data = np.concatenate([patient.iloc[:, begin_met:end_met].fillna(0).values,
                               patient.iloc[:, begin_BA_ratio:end_BA_ratio].fillna(0).values], axis=1)
    cov = [patient.iloc[0]["AGE"], patient.iloc[0]["PTGENDER"], patient.iloc[0]["APOE_e2e4"]]
    long_cov = patient[longitudinal_cov_columns].values.tolist()

    y.append(1)

    static_covariates.append(cov)
    longitudinal_covariates.append(long_cov)
    X.append(met_data)
    rids.append(rid)

# get pMCI patients
for rid, patient in deltaT_pMCI.groupby("RID"):
    met_data = np.concatenate([patient.iloc[:len(delta_t), begin_met:end_met].fillna(0).values,
                               patient.iloc[:len(delta_t), begin_BA_ratio:end_BA_ratio].fillna(0).values], axis=1)
    cov = [patient.iloc[0]["AGE"], patient.iloc[0]["PTGENDER"], patient.iloc[0]["APOE_e2e4"]]
    long_cov = patient[longitudinal_cov_columns].values.tolist()[:len(delta_t)]

    y.append(0)

    static_covariates.append(cov)
    longitudinal_covariates.append(long_cov)
    X.append(met_data)
    rids.append(rid)

X = torch.tensor(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
static_covariates = torch.tensor(static_covariates, dtype=torch.float32)
longitudinal_covariates = torch.tensor(longitudinal_covariates, dtype=torch.float32)
rids = torch.tensor(rids, dtype=torch.float32)

print(X.shape, y.shape, static_covariates.shape, longitudinal_covariates.shape)

if not os.path.exists(f'processed/{cohort}/deltaT'):
    os.makedirs(f'processed/{cohort}/deltaT')

torch.save(X, f'processed/{cohort}/deltaT/X.pt')
torch.save(y, f'processed/{cohort}/deltaT/y.pt')
torch.save(static_covariates, f'processed/{cohort}/deltaT/static_covariates.pt')
torch.save(longitudinal_covariates, f'processed/{cohort}/deltaT/longitudinal_covariates.pt')