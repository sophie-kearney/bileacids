###
# IMPORTS
###

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pandas import read_csv
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

###
# DEFINE CONSTANTS
###

# data structure
input_size = 111  # num features per visit
max_seq = 9      # maximum number of patient visits
output_size = 1

# hyperparameters
num_epochs = 5000
hidden_size = 20
lr = 1e-5

# output of program
eval = True

# columns to use
all_BA = True
LASSO_BA = False
just_BA = False

###
# DATA PROCESSING
###

# data = pd.read_csv("processed/master_data.csv")
data = pd.read_csv("processed/pHCiAD.csv")

# find correct columns
begin_met = data.columns.get_loc("L_HISTIDINE")
end_met = data.columns.get_loc("BUDCA") + 1
begin_BA = data.columns.get_loc("CA")
begin_BA_ratio = data.columns.get_loc("CA_CDCA")
end_BA_ratio = data.columns.get_loc("GLCA_CDCA") + 1

# log 10 scale
# data.iloc[:, begin_met:end_met] = np.log10(data.iloc[:, begin_met:end_met].replace(0, np.nan))
# data.iloc[:, begin_BA_ratio:begin_BA_ratio] = np.log10(data.iloc[:, begin_BA_ratio:begin_BA_ratio].replace(0, np.nan))
# data.fillna(0, inplace=True)
#
# # isoloate HC and AD patients
# pAD = data.groupby("RID").filter(lambda x: (x["DX_VALS"] == 4).all())
# HC = data.groupby("RID").filter(lambda x: (x["DX_VALS"] == 1).all())
# HCAD = pd.concat([pAD, HC], ignore_index=True)

lasso_cols = pd.read_csv("raw/LASSO.csv")["Feature"].to_list()
lasso_cols = lasso_cols[1:]

###
# EXTRACT LONGITUDINAL DATA
###

X = []
y = []
curr_seq = []
curr_rid = data.loc[0, 'RID']
curr_label = 0

for index, row in data.iterrows():
    if all_BA:
        BAs = row[begin_met:end_met].values.tolist()
        BAs += row[begin_BA_ratio:end_BA_ratio].values.tolist()
    elif LASSO_BA:
        BAs = row[lasso_cols].values.tolist()
    elif just_BA:
        BAs = row[begin_BA:end_met].values.tolist()
        BAs += row[begin_BA_ratio:end_BA_ratio].values.tolist()
    else:
        BAs = []

    y_val = row["y"]

    # if RID is the same as the previous row, just add on the label and BAs
    if row['RID'] == curr_rid:
        curr_seq.append(BAs)
        curr_label = y_val
    # if new patient RID in this row
    else:
        # fill up the sequence with zeros if it is less than max_seq
        while len(curr_seq) < max_seq:
            curr_seq.append([0] * input_size)
        X.append(curr_seq)
        y.append(curr_label)

        # reinitialize
        curr_rid = row['RID']
        curr_seq = [BAs]
        curr_label = y_val

# add in the last RID after iterations are complete
while len(curr_seq) < max_seq:
    curr_seq.append([0] * input_size)
X.append(curr_seq)
y.append(curr_label)

# convert nested lists to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

###
# DEFINE MODEL
###

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.5):
        super(SimpleRNN, self).__init__()
        # self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # self.fc = nn.Linear(hidden_size, output_size)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        # h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        # out, _ = self.rnn(x, h0)
        # out = self.fc(out[:, -1, :])
        # return out
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.batch_norm(out[:, -1, :])
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

model = SimpleRNN(input_size, hidden_size, output_size)

###
# TRAIN
###

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs.squeeze(), y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

###
# TEST MODEL
###

if eval:
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        outputs = model(X_test)
        probs = torch.sigmoid(outputs).squeeze()
        predicted_labels = (probs >= 0.5).float()

    accuracy = accuracy_score(y_test.numpy(), predicted_labels.numpy())

    fpr, tpr, thresholds = roc_curve(y_test.numpy(), probs.numpy())
    roc_auc = auc(fpr, tpr)

    print("\n--- PERFORMANCE ---")
    print(f"accuracy: {accuracy:.4f}")
    print(f"roc: {roc_auc:.4f}")
    print("-------------------")

    plt.figure()
    plt.plot(fpr, tpr, color='navy', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()
