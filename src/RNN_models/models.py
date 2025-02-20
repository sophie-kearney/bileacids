import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.5):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x, is_missing_mask):
        x = x * is_missing_mask
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.batch_norm(out[:, -1, :])
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.5):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Helps model know how features change over time
        self.W_mu = torch.nn.Linear(input_size, hidden_size)  # Learns a baseline state (mu) for each feature
        self.W_gamma = torch.nn.Linear(input_size, hidden_size)  # Decay rate (gamma) for each feature

        # Identity mask to restrict interactions to diagonal form
        self.register_buffer("mask", torch.eye(hidden_size))

    def forward(self, x, is_missing_mask):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)

        for t in range(seq_len):
            xt = x[:, t, :]
            combined = torch.cat((xt, h), dim=1)
            zt = torch.sigmoid(self.W_z(combined))
            rt = torch.sigmoid(self.W_r(combined))
            candidate = torch.tanh(self.W_h(torch.cat((xt, rt * h), dim=1)))
            h = (1 - zt) * h + zt * candidate

        out = F.relu(self.fc1(h))
        out = self.fc2(out)
        return out

# class MaskedGRUCell(nn.Module):
#
# class MaskedGRU(nn.Module):
