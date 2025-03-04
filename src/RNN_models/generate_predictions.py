###
# IMPORT PACKAGES
###

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from models import MaskedGRU, RNN
pd.set_option('display.max_columns', None)

###
# SET CONSTANTS
###

cohort = "pMCIiAD" # pHCiAD, pMCIiAD
test_trainval_ratio = .2
train_val_ratio = .2
batch_size = 50
input_size = 111
hidden_size = input_size
num_layers = 3
trained_model = "seed32_MaskedGRU_0.8267"
model_path = f'models/pMCIiAD/{trained_model}'

###
# LOAD DATA
###

X = torch.load(f'processed/{cohort}/X.pt')
y = torch.load(f'processed/{cohort}/y.pt')
is_missing = torch.load(f'processed/{cohort}/is_missing.pt')
time_missing = torch.load(f'processed/{cohort}/time_missing.pt')
static_covariates = torch.load(f'processed/{cohort}/static_covariates.pt')
long_cov = torch.load(f'processed/{cohort}/longitudinal_covariates.pt')

# split into test:train:val
# (X_temp, X_test, y_temp, y_test, mask_temp, mask_test, time_missing_temp, time_missing_test, cov_temp, cov_test, long_cov_temp, long_cov_test) = (
#         train_test_split(X, y, is_missing, time_missing, static_covariates, long_cov, test_size=test_trainval_ratio, random_state=32))
# (X_train, X_val, y_train, y_val, mask_train, mask_val, time_missing_train, time_missing_val, cov_train, cov_val, long_cov_train, long_cov_val) = (
#         train_test_split(X_temp, y_temp, mask_temp, time_missing_temp, cov_temp, long_cov_temp, test_size=train_val_ratio, random_state=32))
#
# # create datasets
# train_dataset = TensorDataset(X_train, y_train, mask_train, time_missing_train, cov_train, long_cov_train)
# val_dataset = TensorDataset(X_val, y_val, mask_val, time_missing_val, cov_val, long_cov_val)
# test_dataset = TensorDataset(X_test, y_test, mask_test, time_missing_test, cov_test, long_cov_test)
#
# # create dataloaders
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

dataset = TensorDataset(X, y, is_missing, time_missing, static_covariates, long_cov)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

###
# LOAD MODEL
###

model = MaskedGRU(input_size, hidden_size, num_classes=2, num_layers=num_layers)
model.load_state_dict(torch.load(model_path))

model.eval()

y_true = []
y_pred = []
all_probs = []
all_covs = []
with torch.no_grad():
    for test_batch in loader:
        X_test_batch, y_test_batch, mask_test_batch, time_missing_test_batch, cov_test_batch, long_cov_test = test_batch
        logits, _ = model(X_test_batch, time_missing_test_batch, mask_test_batch)
        probs = torch.softmax(logits, dim=1)[:, 1]

        all_probs.append(probs.numpy())
        all_covs.append(cov_test_batch)

    all_probs = np.concatenate(all_probs)
    all_covs = np.concatenate(all_covs)

data = pd.DataFrame(all_covs, columns=["rid", "AGE", "PTGENDER", "APOE_e2e4"])
data["ADRiskScore"] = all_probs
data["rid"] = data["rid"].astype(int)

###
# GET TIMES
###

rids = sorted(data["rid"])
master_data = pd.read_csv("processed/master_data.csv").sort_values(by=["RID", "EXAMDATE_RANK"])
times = {}
for rid, patient in master_data.groupby("RID"):
    if int(rid) in rids:
        if 4 in patient["DX_VALS"].values:
            first_ad_row = patient[patient["DX_VALS"] == 4].iloc[0]
            first_ad_viscode2 = int(first_ad_row["VISCODE2"].replace("m","").replace("bl","0"))
            times[rid] = first_ad_viscode2
        else:
            times[rid] = np.nan
data["ADConversionTime"] = data["rid"].map(times)

threshold = data["ADRiskScore"].median()
data["RiskGroup"] = data["ADRiskScore"].apply(lambda x: "High" if x >= threshold else "Low")
data["AD"] = data["ADConversionTime"].notna().astype(int)

###
# GET LONG CONV SCORES
###

hidden_size = 128
dropout = 0.7
input_size = 5
long_cov_model_path = f"models/{cohort}/longCov_seed32_simpleRNN_0.5967"
model = RNN(input_size, hidden_size, 1, num_layers=num_layers, dropout=dropout)
model.load_state_dict(torch.load(long_cov_model_path))

model.eval()
all_probs = []
with torch.no_grad():
    for test_batch in loader:
        X_test_batch, y_test_batch, mask_test_batch, time_missing_test_batch, cov_test_batch, long_cov_test = test_batch

        output = model(long_cov_test)
        probs = torch.sigmoid(output).squeeze()
        all_probs.append(probs.numpy())
all_probs = np.concatenate(all_probs)
data["LongCovRiskScore"] = all_probs

print(data)
data.to_csv(f"processed/{cohort}/{trained_model}_predictions.csv", index=False)