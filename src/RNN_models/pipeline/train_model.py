import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from src.RNN_models.models import RNN, GRU, MaskedGRU

def process_RNN_data(imputed, X, y, is_missing, time_missing, test_trainval_ratio, train_val_ratio, batch_size, seed):
    if imputed:
        (X_temp, X_test, y_temp, y_test, mask_temp, mask_test, time_missing_temp, time_missing_test) = (
            train_test_split(X, y, is_missing, time_missing, test_size=test_trainval_ratio, random_state=seed))

        (X_train, X_val, y_train, y_val, mask_train, mask_val, time_missing_train, time_missing_val) = (
            train_test_split(X_temp, y_temp, mask_temp, time_missing_temp, test_size=train_val_ratio,
                             random_state=seed))

        # create datasets
        train_dataset = TensorDataset(X_train, y_train, mask_train, time_missing_train)
        val_dataset = TensorDataset(X_val, y_val, mask_val, time_missing_val)
        test_dataset = TensorDataset(X_test, y_test, mask_test, time_missing_test)

    else:
        (X_temp, X_test, y_temp, y_test) = train_test_split(X, y, test_size=test_trainval_ratio)
        (X_train, X_val, y_train, y_val) = train_test_split(X_temp, y_temp, test_size=train_val_ratio)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def define_model(model_choice, input_size, hidden_size, num_layers, dropout, lr):
    if model_choice == "simpleRNN":
        model = RNN(input_size, hidden_size, 1, num_layers=num_layers, dropout=dropout)
    elif model_choice == "GRU":
        model = GRU(input_size, hidden_size, 1)
    elif model_choice == "MaskedGRU":
        hidden_size = input_size
        model = MaskedGRU(input_size, hidden_size, num_classes=2, num_layers=num_layers)
    else:
        raise ValueError("Invalid model choice")

    if model_choice == "MaskedGRU":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer

def train_model(model, criterion, optimizer, num_epochs, train_loader, val_loader,
                max_norm, early_stopping, patience, imputed, model_choice):
    losses = []
    val_losses = []
    patience_counter = 0
    best_val_loss = np.inf

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            if imputed:
                X_batch, y_batch, mask_batch, time_missing_batch = batch
            else:
                X_batch, y_batch = batch
                # prevent unbound variables
                time_missing_batch, mask_batch = None, None

            if model_choice == "MaskedGRU":
                logits, h_c = model(X_batch, time_missing_batch, mask_batch)
                loss = criterion(logits, y_batch.long())
            else:
                output = model(X_batch)
                loss = criterion(output.squeeze(1), y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_batch in val_loader:
                if imputed:
                    X_val_batch, y_val_batch, mask_val_batch, time_missing_val_batch = val_batch
                else:
                    X_val_batch, y_val_batch = val_batch
                if model_choice == "MaskedGRU":
                    val_logits, _ = model(X_val_batch, time_missing_val_batch, mask_val_batch)
                    loss = criterion(val_logits, y_val_batch.long())
                else:
                    val_output = model(X_val_batch)
                    loss = criterion(val_output.squeeze(1), y_val_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if train_loss > 4 or val_loss > 4:
            raise ValueError("Loss too high. Start over")

        if (epoch + 1) % 10 == 0:
            print(f'  Epoch [{epoch + 1}/{num_epochs}], Test Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if early_stopping and patience_counter >= patience:
            break

    plt.figure()
    plt.plot(range(len(losses)), losses, label='Training Loss')
    plt.plot(range(len(losses)), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'BARS {model_choice} Training')
    plt.legend()
    plt.show()
    return model

def define_long_model(input_size, hidden_size, num_layers, dropout, lr):
    model = RNN(input_size, hidden_size, 1, num_layers=num_layers, dropout=dropout)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer

def train_long_model(model, criterion, optimizer, X, y, seed, long_patience):
    # --- set local hyperparameters, these are mostly unchanged
    test_trainval_ratio = 0.2
    batch_size = 50
    num_epochs = 500
    max_norm = 0.5

    # --- split data
    (X_temp, X_test, y_temp, y_test) = train_test_split(X, y, test_size=test_trainval_ratio, random_state=seed)
    (X_train, X_val, y_train, y_val) = train_test_split(X_temp, y_temp, test_size=test_trainval_ratio, random_state=seed)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- train model

    losses = []
    val_losses = []
    patience_counter = 0
    best_val_loss = np.inf

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            X_batch, y_batch = batch

            output = model(X_batch)
            loss = criterion(output.squeeze(1), y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_batch in val_loader:
                X_val_batch, y_val_batch = val_batch
                val_output = model(X_val_batch)
                loss = criterion(val_output.squeeze(1), y_val_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'  Epoch [{epoch + 1}/{num_epochs}], Test Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= long_patience:
            break

    plt.figure()
    plt.plot(range(len(losses)), losses, label='Training Loss')
    plt.plot(range(len(losses)), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Longitudinal Covariates RNN Training')
    plt.legend()
    plt.show()

    return model, test_loader