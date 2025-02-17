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

##
# DEFINE CONSTANTS
###

# data structure
input_size = 112  # num features per visit
output_size = 1

# hyperparameters
num_epochs = 35000
hidden_size = 20
lr = 1e-5

# output of program
eval = True

###
# DATA PROCESSING
###

X = torch.load('processed/X.pt')
y = torch.load('processed/y.pt')

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

eval = True
losses = []
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
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
    # model.eval()

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