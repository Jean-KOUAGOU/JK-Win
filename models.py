import torch, torch.nn as nn, numpy as np
import torch.nn.functional as F
from modules import *

class LSTM(nn.Module):
    def __init__(self, input_dim, proj_dim, n_layers):
        super().__init__()
        self.name = "LSTM"
        self.proj_dim = proj_dim
        self.n_layers = n_layers
        self.loss = nn.MSELoss()
        self.lstm = nn.LSTM(input_dim//2, self.proj_dim, self.n_layers, batch_first=True)
        self.bn = nn.BatchNorm1d(self.proj_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(self.proj_dim, self.proj_dim)
        self.fc2 = nn.Linear(self.proj_dim, self.proj_dim)
        self.fc3 = nn.Linear(self.proj_dim, 9)
        
    def forward(self, x):
        seq, _ = self.lstm(torch.cat(x.chunk(2, 2), 1))
        out = seq.sum(1).view(-1, self.proj_dim)
        x = self.dropout1(F.gelu(self.fc1(out)))
        x = self.dropout2(x + self.fc2(x))
        x = self.bn(x)
        x = self.fc3(x).reshape(-1, 9)
        return x

        
class GRU(nn.Module):
    def __init__(self, input_dim, proj_dim, n_layers):
        super().__init__()
        self.name = "GRU"
        self.proj_dim = proj_dim
        self.n_layers = n_layers
        self.loss = nn.MSELoss()
        self.gru = nn.GRU(input_dim//2, self.proj_dim, self.n_layers, batch_first=True)
        self.bn = nn.BatchNorm1d(self.proj_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(self.proj_dim, self.proj_dim)
        self.fc2 = nn.Linear(self.proj_dim, self.proj_dim)
        self.fc3 = nn.Linear(self.proj_dim, 9)
    
    def forward(self, x):
        seq, _ = self.gru(torch.cat(x.chunk(2, 2), 1))
        out = seq.sum(1).view(-1, self.proj_dim)
        x = self.dropout1(F.gelu(self.fc1(out)))
        x = self.dropout2(x + self.fc2(x))
        x = self.bn(x)
        x = self.fc3(x).reshape(-1, 9)
        return x
    
    
class MLP(nn.Module):
    def __init__(self, input_dim, proj_dim):
        super().__init__()
        self.name = "MLP"
        self.proj_dim = proj_dim
        self.input_dim = input_dim
        self.loss = nn.MSELoss()
        self.bn = nn.BatchNorm1d(self.proj_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(self.input_dim, self.proj_dim)
        self.fc2 = nn.Linear(self.proj_dim, self.proj_dim)
        self.fc3 = nn.Linear(self.proj_dim, self.proj_dim)
        self.fc4 = nn.Linear(self.proj_dim, 9)
    
    def forward(self, x):
        x = self.dropout1(self.fc1(x))
        x = F.gelu(x + self.fc2(x))
        x = self.dropout2(x + self.fc3(x))
        x = self.bn(x.view(-1, self.proj_dim))
        x = self.fc4(x).reshape(-1, 9)
        return x

    
class SetTransformer(nn.Module):
    def __init__(self, input_dim, proj_dim, num_inds, num_heads, num_seeds):
        super(SetTransformer, self).__init__()
        self.name = "SetTransformer"
        self.proj_dim = proj_dim
        self.num_inds = num_inds
        self.num_heads = num_heads
        self.num_seeds = num_seeds
        self.dropout = nn.Dropout(0.2)
        self.loss = nn.MSELoss()
        self.enc = nn.Sequential(
                ISAB(input_dim, self.proj_dim, self.num_heads, self.num_inds),
                ISAB(self.proj_dim, self.proj_dim, self.num_heads, self.num_inds))
        self.dec = nn.Sequential(
                PMA(self.proj_dim, self.num_heads, self.num_seeds),
                nn.Linear(self.proj_dim, 9))

    def forward(self, x):
        x = self.dropout(self.enc(x))
        x = self.dec(x).reshape(-1, 9)
        return x