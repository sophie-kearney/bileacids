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
from models import RNN, GRU

cohort = "pHCiAD"    # pMCIiAD pCHiAD
model_choice = "simpleRNN" # GRU, simpleRNN, MaskedGRU

###
# DATA PROCESSING
###

X = torch.load(f'processed/{cohort}/X.pt')
y = torch.load(f'processed/{cohort}/y.pt')
is_missing = torch.load(f'processed/{cohort}/is_missing.pt')
time_missing = torch.load(f'processed/{cohort}/time_missing.pt')

(X_train, X_test, y_train, y_test, mask_train, mask_test, time_missing_test,
    time_missing_train) = train_test_split(X, y, is_missing, time_missing,
                                           test_size=0.2, random_state=42)

##
# DEFINE CONSTANTS
###

# data structure
input_size = X.shape[2]  # num features per visit
output_size = 1

# hyperparameters
num_epochs = 35000
hidden_size = input_size
lr = 1e-5

# output of program
eval = True

###
# DEFINE MODEL
###

if model_choice == "simpleRNN":
    model = RNN(input_size, hidden_size, output_size)
elif model_choice == "GRU":
    model = GRU(input_size, hidden_size, output_size)
# elif model_choice == "MaskedGRU":
#     model = MaskedGRU(input_size, hidden_size, output_size)
else:
    raise ValueError("Invalid model choice")

###
# TRAIN
###

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

losses = []
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train, mask_train)
    loss = criterion(outputs.squeeze(), y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

plt.figure()
plt.plot(range(num_epochs), losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()

###
# TEST MODEL
###

if eval:
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        outputs = model(X_test, mask_test)
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