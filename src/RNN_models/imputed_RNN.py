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
from models import RNN, GRU, MaskedGRU

##
# DEFINE CONSTANTS
###

# hyperparameters
num_epochs = 10000
lr = 1e-5
patience = 100
test_trainval_ratio = .2
train_val_ratio = .6

# program parameters
cohort = "pHCiAD"          # pMCIiAD pCHiAD
model_choice = "simpleRNN" # GRU, simpleRNN, MaskedGRU
eval = True

###
# DATA PROCESSING
###

X = torch.load(f'processed/{cohort}/X.pt')
y = torch.load(f'processed/{cohort}/y.pt')
is_missing = torch.load(f'processed/{cohort}/is_missing.pt')
time_missing = torch.load(f'processed/{cohort}/time_missing.pt')

# (X_train, X_test, y_train, y_test, mask_train, mask_test, time_missing_train, time_missing_test) = (
#     train_test_split(X, y, is_missing, time_missing, test_size=0.2, random_state=42))

(X_train, X_temp, y_train, y_temp, mask_train, mask_temp, time_missing_train, time_missing_temp) = (
    train_test_split(X, y, is_missing, time_missing, test_size=test_trainval_ratio, random_state=42))

(X_val, X_test, y_val, y_test, mask_val, mask_test, time_missing_val, time_missing_test) = (
    train_test_split(X_temp, y_temp, mask_temp, time_missing_temp, test_size=train_val_ratio, random_state=42))

# data structure
input_size = X.shape[2]  # num features per visit
output_size = 1
hidden_size = input_size

###
# DEFINE MODEL
###

if model_choice == "simpleRNN":
    model = RNN(input_size, hidden_size, output_size)
elif model_choice == "GRU":
    model = GRU(input_size, hidden_size, output_size)
elif model_choice == "MaskedGRU":
    model = MaskedGRU(input_size, hidden_size, num_classes=2, num_layers=1)
else:
    raise ValueError("Invalid model choice")

###
# TRAIN
###

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

losses = []
val_losses = []
patience_counter = 0
best_val_loss = np.inf
for epoch in range(num_epochs):
    model.train()

    if model_choice == "MaskedGRU":
        logits, h_c = model(X_train, time_missing_train, mask_train)
        loss = criterion(logits, y_train.long())
    else:
        output = model(X_train, mask_train)
        loss = criterion(output.squeeze(), y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        if model_choice == "MaskedGRU":
            val_logits, _ = model(X_val, time_missing_val, mask_val)
            val_loss = criterion(val_logits, y_val.long())
        else:
            val_output = model(X_val, mask_val)
            val_loss = criterion(val_output.squeeze(), y_val)

        val_losses.append(val_loss.item())
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch + 1}')
        break

plt.figure()
plt.plot(range(len(losses)), losses, label='Training Loss')
plt.plot(range(len(losses)), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
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
        if model_choice == "MaskedGRU":
            logits, _ = model(X_test, time_missing_test, mask_test)
            probs = logits[:, 1]
            predicted_labels = torch.argmax(logits, dim=1)
        else:
            output = model(X_test, mask_test)
            probs = torch.sigmoid(output).squeeze()
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