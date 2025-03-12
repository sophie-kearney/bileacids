###
# IMPORT PACKAGES
###

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from models import MaskedGRU, RNN
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
# from EmbeddingExtractor import EmbeddingExtractor
import random

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
# trained_model = "42_MaskedGRU_03121122_0.8961"
trained_model = "41_MaskedGRU_03121130_0.7963"
model_path = f'models/pMCIiAD/{trained_model}'

seed = 123
random.seed = seed
np.random.seed = seed
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

###
# LOAD MODEL
###

model = MaskedGRU(input_size, hidden_size, num_classes=2, num_layers=num_layers)
model.load_state_dict(torch.load(model_path))

model.eval()

###
# LOAD DATA
###

X = torch.load(f'processed/{cohort}/X.pt')
y = torch.load(f'processed/{cohort}/y.pt')
is_missing = torch.load(f'processed/{cohort}/is_missing.pt')
time_missing = torch.load(f'processed/{cohort}/time_missing.pt')
static_covariates = torch.load(f'processed/{cohort}/static_covariates.pt')
long_cov = torch.load(f'processed/{cohort}/longitudinal_covariates.pt')
rids = torch.load(f'processed/{cohort}/rids.pt')

feature_names = pd.read_csv(f'processed/feature_names.csv')

dataset = TensorDataset(X, static_covariates, y, time_missing, is_missing, rids)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

###
# GET CONTRIBUTIONS
###

batch_size = 586
hidden_size = 111
class_size = 2
seq_size = 13

model.eval()
with torch.no_grad():
    W_o = model.fc.weight # (num_classes, hidden_size) = (2, 111)
    b_o = model.fc.bias   # (num_classes,) = (2,)
    outputs, h_t = model(X, time_missing, is_missing) # (batch_size, seq_len, num_classes) = (586, 13, 2)
    # print(f"h_t shape: {h_t.shape}")

    c_t_d = [] # (hidden_size, seq_size, batch_size) = (111, 13, 586)
    for d in range(hidden_size):
        W_o_d = W_o[1, d] # get the weight of the positive class
        b_o_d = b_o[1]    # get the bias of the positive class

        c_d = []
        for t in range(seq_size):
            h_d_t = h_t[:, t, d] # (batch_size,) = (586,)
            c_d.append(W_o_d * h_d_t + b_o_d)

        c_t_d.append(c_d)

    # print(f"c_t_d shape: ({len(c_t_d)}, {len(c_t_d[0])}, {len(c_t_d[0][1])})")
    u_d = np.mean(c_t_d, axis=1)       # (hidden_size, seq_size, batch_size) -> (hidden_size, batch_size)
                                       # (111, 13, 586) -> (111, 586)
    # print("u_d shape: ", u_d.shape)

    u_d_abs = np.abs(u_d)
    mean_abs_u_d = np.mean(u_d_abs, axis=1)    # (hidden_size, batch_size) -> (hidden_size,)
                                               # (111, 586) -> (111,)
    # print("mean_abs_u_d shape: ", mean_abs_u_d.shape)

importance = pd.DataFrame({
        'feature_ndx': np.arange(111),
        'contribution': mean_abs_u_d
})

feature_names = pd.read_csv("processed/feature_names.csv")
importance = importance.merge(feature_names, on='feature_ndx')
importance = importance.sort_values(by='contribution', ascending=False)
print(importance.head(20))
importance.to_csv(f"processed/{trained_model}importance.csv", index=False)
# print(f"{trained_model}importance.csv")

top_20_features = importance.sort_values(by='contribution', ascending=False).head(20)

plt.figure(figsize=(8, 6))
plt.barh(top_20_features['bileacid'], top_20_features['contribution'], color='skyblue', align='center')
plt.xlabel('Contribution')
plt.ylabel('Feature')
plt.title('Top 20 Features by Contribution')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()