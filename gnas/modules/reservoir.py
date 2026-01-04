import torch
import torch.nn as nn
import torch.nn.functional as F
from search_config import args

class Reservoir(nn.Module):
    def __init__(self, reservoir_size):
        super(Reservoir, self).__init__()
        self.reservoir_size = reservoir_size
        self.spectral_radius = args.spectral_radius
        self.sparsity = args.sparsity
        self.register_buffer("W_in", torch.empty(0))
        self.register_buffer("W_res", torch.empty(0))
        self.initialized = False

    def initialize_weights(self, input_size, device):
        W_in = torch.randn(self.reservoir_size, input_size, device=device) * 0.1
        W_res = torch.randn(self.reservoir_size, self.reservoir_size, device=device)
        mask = torch.rand_like(W_res) > self.sparsity
        W_res[mask] = 0.0

        eigenvalues = torch.linalg.eigvals(W_res)
        max_eigenvalue = torch.max(torch.abs(eigenvalues)).real
        if max_eigenvalue > 0:
            W_res *= self.spectral_radius / max_eigenvalue
        
        self.W_in = W_in
        self.W_res = W_res
        self.initialized = True

    def forward(self, x):
        batch_size, seq_len, input_size = x.size()
        device = x.device

        if not self.initialized:
           self.initialize_weights(input_size, device)

        with torch.no_grad():
            states = torch.empty(batch_size, seq_len, self.reservoir_size, device=device)
            state = torch.zeros(batch_size, self.reservoir_size, device=device)

            for t in range(seq_len):
                u = x[:, t, :]
                state = torch.tanh(u @ self.W_in.T + state @ self.W_res.T)
                states[:, t, :] = state
        
        return states


class EchoStateNetwork(nn.Module):
    def __init__(self, out_channels):
        super(EchoStateNetwork, self).__init__()

        reservoir_size = args.reservoir_size
        self.reservoir = Reservoir(reservoir_size)

        self.readout = nn.Linear(reservoir_size, out_channels)

    def forward(self, x):

        x = x.permute(0, 2, 1)

        reservoir_out = self.reservoir(x)

        output = self.readout(reservoir_out)
        
        return output.permute(0, 2, 1)




class ESN(nn.Module):
    def __init__(self, out_channels):
        super(ESN, self).__init__()
        self.esn = EchoStateNetwork(out_channels)

    def forward(self, x):
        x = self.esn(x)
        return x

# A operação para uma camada LSTM simples
class LSTM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=out_channels, batch_first=True)
        self.out_channels = out_channels

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_permuted)
        return lstm_out.permute(0, 2, 1)