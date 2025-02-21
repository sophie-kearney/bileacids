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
num_epochs = 5000
lr = .00001
output_size = 1
hidden_size = 64
patience = 150
test_trainval_ratio = .2
train_val_ratio = .25
dropout = 0.5
num_layers = 5

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

(X_temp, X_test, y_temp, y_test, mask_temp, mask_test, time_missing_temp, time_missing_test) = (
    train_test_split(X, y, is_missing, time_missing, test_size=test_trainval_ratio))

(X_train, X_val, y_train, y_val, mask_train, mask_val, time_missing_train, time_missing_val) = (
    train_test_split(X_temp, y_temp, mask_temp, time_missing_temp, test_size=train_val_ratio))

# data structure
input_size = X.shape[2]  # num features per visit

# create datasets
train_dataset = TensorDataset(X_train, y_train, mask_train, time_missing_train)
val_dataset = TensorDataset(X_val, y_val, mask_val, time_missing_val)
test_dataset = TensorDataset(X_test, y_test, mask_test, time_missing_test)

# create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

###
# DEFINE MODEL
###

if model_choice == "simpleRNN":
    model = RNN(input_size, hidden_size, 1, num_layers=num_layers, dropout=dropout)
elif model_choice == "GRU":
    model = GRU(input_size, hidden_size, output_size)
elif model_choice == "MaskedGRU":
    model = MaskedGRU(input_size, input_size, num_classes=2, num_layers=1)
else:
    raise ValueError("Invalid model choice")
print(model)

###
# TRAIN
###

if model_choice == "MaskedGRU":
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=lr)

losses = []
val_losses = []
patience_counter = 0
best_val_loss = np.inf

for epoch in range(num_epochs):
    model.train()

    train_loss = 0
    for batch in train_loader:
        X_batch, y_batch, time_missing_batch, mask_batch = batch
        if model_choice == "MaskedGRU":
            logits, h_c = model(X_batch, time_missing_batch, mask_batch)
            loss = criterion(logits, y_batch.long())
        else:
            output = model(X_batch, mask_batch)
            probabilities = torch.sigmoid(output).squeeze(1)
            loss = criterion(output.squeeze(1), y_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    losses.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_batch in val_loader:
            X_val_batch, y_val_batch, time_missing_val_batch, mask_val_batch = val_batch
            if model_choice == "MaskedGRU":
                val_logits, _ = model(X_val_batch, time_missing_val_batch, mask_val_batch)
                loss = criterion(val_logits, y_val_batch.long())
            else:
                val_output = model(X_val_batch, mask_val_batch)
                loss = criterion(val_output.squeeze(1), y_val_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)


    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
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
    all_probs = []

    with torch.no_grad():
        for test_batch in test_loader:
            X_test_batch, y_test_batch, time_missing_test_batch, mask_test_batch = test_batch
            if model_choice == "MaskedGRU":
                logits, _ = model(X_test_batch, time_missing_test_batch, mask_test_batch)
                probs = logits[:, 1]
                predicted_labels = torch.argmax(logits, dim=1)
            else:
                output = model(X_test_batch, mask_test_batch)
                probs = torch.sigmoid(output).squeeze()
                predicted_labels = (probs >= 0.5).float()

            y_true.append(y_test_batch.numpy())
            y_pred.append(predicted_labels.numpy())
            all_probs.append(probs.numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    all_probs = np.concatenate(all_probs)

    accuracy = accuracy_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, all_probs)
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