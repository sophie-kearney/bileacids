import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.5):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x, is_missing_mask):
        # x = x * is_missing_mask
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

class MaskedGRUCell(torch.nn.Module):
    """
    Standard GRU Gates (r_t, z_t) regulate memory updates.
    r_t: Reset gate. Controls how much of previous hidden state gets used.
    z_t: Update gate. Controls a mix between previous and candidate (current) hidden state.
    Decay Mechanism ensures that missing values gradually decay toward a baseline over time.
    Observation Indicator (m_t) controls whether decay applies or not.
    """

    def __init__(self, input_size, hidden_size):
        super(MaskedGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size  # hidden_size should match input_size

        # Input-hidden weights
        self.Wir = torch.nn.Linear(input_size, hidden_size)
        self.Wiz = torch.nn.Linear(input_size, hidden_size)
        self.Win = torch.nn.Linear(input_size, hidden_size)

        # Hidden-hidden weights
        self.Whr = torch.nn.Linear(hidden_size, hidden_size, bias=False)  # Reset gate
        self.Whz = torch.nn.Linear(hidden_size, hidden_size, bias=False)  # Update gate
        self.Whn = torch.nn.Linear(hidden_size, hidden_size, bias=False)  # New hidden state

        # Time-based weight matrices
        self.W_mu = torch.nn.Linear(input_size, hidden_size)  # Learns a baseline state (mu) for each feature
        self.W_gamma = torch.nn.Linear(input_size, hidden_size)  # Decay rate (gamma) for each feature

        # Identity mask to restrict interactions to diagonal form
        self.register_buffer("mask", torch.eye(hidden_size))
        self.apply_mask()

    def apply_mask(self):
        """Applies the identity mask to weight matrices to enforce diagonal structure."""
        with torch.no_grad():
            self.Wir.weight.data *= self.mask
            self.Wiz.weight.data *= self.mask
            self.Win.weight.data *= self.mask
            self.Whr.weight.data *= self.mask
            self.Whz.weight.data *= self.mask
            self.Whn.weight.data *= self.mask

    def forward(self, x_t, h_t_minus_1, delta_t, m_t):
        """
        x_t: Feature values at time t
        h_t_minus_1: Previous hidden state
        delta_t: Elapsed time since last observation
        m_t: Observation indicator (1=observed, 0=missing)
        """
        # Reset gate (r_t) and Update gate (z_t) with identity mask applied
        r_t = torch.sigmoid((self.Wir(x_t)) + (self.Whr(h_t_minus_1)))
        z_t = torch.sigmoid((self.Wiz(x_t)) + (self.Whz(h_t_minus_1)))

        # Candidate activation (n_t): Represents new info to be stored in the cell state from reset_gate
        n_t = torch.tanh((self.Win(x_t)) + (self.Whn(h_t_minus_1 * r_t)))

        # Update cell state using z_t (old info) and n_t (new info)
        h_t = (1 - z_t) * h_t_minus_1 + z_t * n_t

        # Models how features change over time when NOT observed
        mu_t = self.W_mu(x_t)  # Learns a baseline value for each feature
        gamma_t = self.W_gamma(delta_t)  # Decay rate

        # Compute hidden state based on elapsed time and observation by applying exponential decay to hidden state
        # if m_t=1, no decay applied. Else hidden state decays towards baseline
        h_hat_t = mu_t + (h_t - mu_t) * torch.exp(-gamma_t * m_t)

        # Return updated hidden state and cell state
        return h_hat_t

class MaskedGRU(torch.nn.Module):
    """
    Uses multiple GRU layers to capture sequential patterns in the data.
    Incorporates elapsed time (delta_t) and observation indicators (m_t) to handle missing values properly.
    Outputs a prediction (logits) at the final time step.
    """

    def __init__(self, input_size, hidden_size, num_classes=2, num_layers=1):
        super(MaskedGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Custom GRU cells - creates multiple layers
        # First layer takes input_size; subsequent layers take hidden_size
        # Stacking multiple GRU layers helps the model learn hierarchical time dependencies.
        self.gru_cells = torch.nn.ModuleList(
            [MaskedGRUCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]
        )

        # Output layer. Maps final hidden state to class probabilities
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x, delta_t, m_t):
        """
        x: Input features (batch_size, seq_len, input_size)
        delta_t: Elapsed time since last observation (batch_size, seq_len, input_size)
        m_t: Observation indicators (batch_size, seq_len, input_size)
        """
        # Initialize hidden and cell states
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        # c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Process each time step
        for t in range(seq_len):
            x_t = x[:, t, :]
            delta_t_t = delta_t[:, t, :]
            m_t_t = m_t[:, t, :]

            # process each GRU layer
            for layer in self.gru_cells:
                h_t = layer(x_t, h_t, delta_t_t, m_t_t)

        # Final prediction at the last time step
        logits = self.fc(h_t)
        return logits, h_t