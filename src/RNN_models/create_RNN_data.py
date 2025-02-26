###
# IMPORTS
###

import numpy as np
import pandas as pd
import torch
import sys, os

###
# DEFINE CONSTANTS
###

num_mets = 111
# cohort = "pMCIiAD" # pHCiAD, pMCIiAD
cohort = os.getenv('COHORT', 'pMCIiAD')
imputed = os.getenv('IMPUTED', 'True').lower() in ('true', '1', 't')
k_env = os.getenv('K')
k = int(k_env) if k_env else None

print(k, imputed, cohort)

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

# remove infinite values and NAs with 0 in the ratio columns, these are coming from dividing by 0
# pHCiAD[begin_BA_ratio:end_BA_ratio] = pHCiAD[begin_BA_ratio:end_BA_ratio].replace([np.inf, -np.inf, np.nan], 0)

###
# IMPUTATION
###

if imputed:
    all_visits = sorted(pHCiAD["VISCODE2"].unique(), key=lambda x: (x != 'bl', int(x[1:]) if x[1:].isdigit() else float('inf')))

    if k is not None:
        all_visits = all_visits[:k]

    visit_mapping = {visit: (0 if visit == 'bl' else int(visit[1:])) for visit in pHCiAD["VISCODE2"].unique()}

    X = []
    y = []
    is_missing = []
    time_missing = []
    rids = []
    static_covariates = []

    for rid, patient in pHCiAD.groupby("RID"):
        curr_seq = []    # current sequence of visits for one patient
        no_bl = False    # flag for if patient has missing visits before any are filled
        missingness = [] # length of attributes, 1 if filled, 0 if missing
        time_miss = []   # length of attributes, 0 if no time since last and delta time if there is

        for visit in all_visits:
            # patient has data for that visit
            if visit in patient["VISCODE2"].values:
                row = patient[patient["VISCODE2"] == visit]
                met_data = [float(x) for x in row.iloc[0, begin_met:end_met].values.tolist()] + \
                           [float(x) for x in row.iloc[0, begin_BA_ratio:end_BA_ratio].values.tolist()]

                # if there are missing values, forward fill and track in missingness
                curr_miss = []
                curr_time = []
                for i in range(len(met_data)):
                    if np.isnan(met_data[i]):
                        # forward fill data from the last visit to this visit
                        if curr_seq != []:
                            met_data[i] = curr_seq[-1][i]
                        else:
                            met_data[i] = 0 # TODO - this is a placeholder, we should use a better imputation method
                        curr_miss.append(0)

                        if visit != 'bl':
                            # get change in time since last filled in visit
                            prev_visit = -1
                            while (prev_visit + len(curr_seq) > 0) and (missingness[prev_visit][i] == 0):
                                prev_visit -= 1
                            prev_visit += len(curr_seq) # update prev_visit so we are counting from the left again and not the right
                            last = visit_mapping[all_visits[prev_visit]]
                            cur = visit_mapping[visit]
                            diff = cur-last
                        else:
                            diff = 0
                        curr_time.append(diff)

                    else:
                        curr_miss.append(1)
                        curr_time.append(0)

                missingness.append(curr_miss)
                curr_seq.append(met_data)
                time_miss.append(curr_time)

            # the patient doesn't have data for that visit
            else:
                if curr_seq != []:
                    # get change in time since last filled in visit
                    prev_visit = -1
                    while missingness[prev_visit][0] == 0:
                        prev_visit -= 1
                    prev_visit += len(curr_seq)
                    last = visit_mapping[all_visits[prev_visit]]
                    cur = visit_mapping[visit]
                    diff = cur-last
                    time_miss.append([diff] * len(curr_seq[-1]))

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
            time_missing.append(time_miss)

            cov = [rid, patient.iloc[0]["AGE"], patient.iloc[0]["PTGENDER"], patient.iloc[0]["APOE_e2e4"]]
            static_covariates.append(cov) # TODO - there are NA values in the covariates

            # add the classification to y
            if 4 in patient["DX_VALS"].values:
                y.append(1)
            else:
                y.append(0)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    is_missing = torch.tensor(is_missing, dtype=torch.float32)
    time_missing = torch.tensor(time_missing, dtype=torch.float32)
    static_covariates = torch.tensor(static_covariates, dtype=torch.float32)

    # print(X.shape)
    # print(y.shape)
    # print(is_missing.shape)
    # print(time_missing.shape)

    # no NAs, no Infs
    X = torch.tensor(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32)
    # print("                         IS NA         IS INF")
    # print("X                 ", torch.isnan(X).any(), torch.isinf(X).any())
    # print("y                 ", torch.isnan(y).any(), torch.isinf(y).any())
    # print("is_missing        ", torch.isnan(is_missing).any(), torch.isinf(is_missing).any())
    # print("time_missing      ", torch.isnan(time_missing).any(), torch.isinf(time_missing).any())
    # print("static_covariates ", torch.isnan(time_missing).any(), torch.isinf(time_missing).any())

    if not os.path.exists(f'processed/{cohort}'):
        os.makedirs(f'processed/{cohort}')

    torch.save(X, f'processed/{cohort}/X.pt')
    torch.save(y, f'processed/{cohort}/y.pt')
    torch.save(is_missing, f'processed/{cohort}/is_missing.pt')
    torch.save(time_missing, f'processed/{cohort}/time_missing.pt')
    torch.save(static_covariates, f'processed/{cohort}/static_covariates.pt')

else:
    X = []
    y = []
    rids = []
    static_covariates = []

    for rid, patient in pHCiAD.groupby("RID"):
        met_data = np.concatenate([patient.iloc[:, begin_met:end_met].fillna(0).values,
                                   patient.iloc[:, begin_BA_ratio:end_BA_ratio].fillna(0).values], axis=1)
        cov = [patient.iloc[0]["AGE"], patient.iloc[0]["PTGENDER"], patient.iloc[0]["APOE_e2e4"]]

        if 4 in patient["DX_VALS"].values:
            y.append(1)
        else:
            y.append(0)

        static_covariates.append(cov)
        X.append(met_data)
        rids.append(rid)

    X = torch.nn.utils.rnn.pad_sequence([torch.tensor(x, dtype=torch.float32) for x in X], batch_first=True)
    X = torch.tensor(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32)

    y = torch.tensor(y, dtype=torch.float32)
    static_covariates = torch.tensor(static_covariates, dtype=torch.float32)
    rids = torch.tensor(rids, dtype=torch.float32)

    if not os.path.exists(f'processed/{cohort}/not_imputed'):
        os.makedirs(f'processed/{cohort}/not_imputed')

    torch.save(X, f'processed/{cohort}/not_imputed/X.pt')
    torch.save(y, f'processed/{cohort}/not_imputed/y.pt')
    torch.save(static_covariates, f'processed/{cohort}/not_imputed/static_covariates.pt')