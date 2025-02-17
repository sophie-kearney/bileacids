###
# IMPORTS
###

import numpy as np
import pandas as pd
import pickle
import torch
import sys

###
# DEFINE CONSTANTS
###

num_mets = 111

###
# PARSE DATA
###

data = pd.read_csv("processed/master_data.csv")

# find correct columns
begin_met = data.columns.get_loc("L_HISTIDINE")
end_met = data.columns.get_loc("BUDCA") + 1
begin_BA = data.columns.get_loc("CA")
begin_BA_ratio = data.columns.get_loc("CA_CDCA")
end_BA_ratio = data.columns.get_loc("GLCA_CDCA") + 1

# log 10 scale
data.iloc[:, begin_met:end_met] = np.log10(data.iloc[:, begin_met:end_met].replace(0, np.nan))
data.iloc[:, begin_BA_ratio:begin_BA_ratio] = np.log10(data.iloc[:, begin_BA_ratio:begin_BA_ratio].replace(0, np.nan))

# isoloate pHC patients
pHC = data.groupby("RID").filter(lambda x: (x["DX_VALS"] == 1).all())

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

# remove specific cases that fail previous filtering
iAD = data[data["RID"].isin(rids)]

pHCiAD = pd.concat([pHC, iAD], ignore_index=True)

# remove infinite values
numeric_cols = pHCiAD.select_dtypes(include=[np.number]).columns
pHCiAD[numeric_cols] = np.where(np.isinf(pHCiAD[numeric_cols]), 0, pHCiAD[numeric_cols])
# pHCiAD.to_csv("processed/pHCiAD.csv", index=False)

###
# IMPUTATION
###

all_visits = pHCiAD["VISCODE2"].unique()
X = []
y = []
is_missing = []

for rid, patient in pHCiAD.groupby("RID"):
    curr_seq = []    # current sequence of visits for one patient
    no_bl = False    # flag for if patient has missing visits before any are filled
    missingness = [] # length of attributes, 1 if filled, 0 if missing
    for visit in all_visits:
        if visit in patient["VISCODE2"].values:
            row = patient[patient["VISCODE2"] == visit]
            met_data = [float(x) for x in row.iloc[0, begin_met:end_met].values.tolist()] + \
                       [float(x) for x in row.iloc[0, begin_BA_ratio:end_BA_ratio].values.tolist()]

            # if there are missing values, forward fill and track in missingness
            curr_miss = []
            for i in range(len(met_data)):
                if np.isnan(met_data[i]):
                    # forward fill data from the last visit to this visit
                    if curr_seq != []:
                        met_data[i] = curr_seq[-1][i]
                    else:
                        met_data[i] = 0 # TODO - this is a placeholder, we should use a better imputation method
                    curr_miss.append(0)
                else:
                    curr_miss.append(1)

            missingness.append(curr_miss)
            curr_seq.append(met_data)
        else:
            if curr_seq != []:
                # forward fill imputation from last curr_seq
                curr_seq.append(curr_seq[-1])
                missingness.append([0] * len(curr_seq[-1])) # we have to fill every value so it is all 0
            else:
                # patient has missing visits before any are filled, does not have baseline data
                no_bl = True
                continue

    # if the patient did have a baseline value, add it to our X
    if not no_bl:
        X.append(curr_seq)
        is_missing.append(missingness)

        # add the classification to y
        if 4 in patient["DX_VALS"].values:
            y.append(1)
        else:
            y.append(0)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
is_missing = torch.tensor(is_missing, dtype=torch.float32)

# no NAs, no Infs
# print(torch.isnan(X).any(), torch.isinf(X).any())
# print(torch.isnan(y).any(), torch.isinf(y).any())
# print(torch.isnan(is_missing).any(), torch.isinf(is_missing).any())

torch.save(X, 'processed/RNN/X.pt')
torch.save(y, 'processed/RNN/y.pt')
torch.save(is_missing, 'processed/RNN/is_missing.pt')