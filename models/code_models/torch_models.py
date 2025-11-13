import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class BaseNeuralNetwork(nn.Module):
    
    def __init__(self, input_dim: int, output_dim: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, x):
        raise NotImplementedError

class SimpleMLP(BaseNeuralNetwork):
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], 
                 output_dim: int = 3, dropout: float = 0.2):
        super().__init__(input_dim, output_dim)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
class DeepMLP(BaseNeuralNetwork):
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64, 32], 
                 output_dim: int = 3, dropout: float = 0.3):
        super().__init__(input_dim, output_dim)
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(self, x):
        x = self.input_layer(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
        
        return self.output_layer(x)


class ResidualBlock(nn.Module):
    
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += residual
        return F.relu(out)


class ResNet(BaseNeuralNetwork):
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_blocks: int = 3, output_dim: int = 3, dropout: float = 0.2):
        super().__init__(input_dim, output_dim)
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.input_layer(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        return self.output_layer(x)


class AttentionMLP(BaseNeuralNetwork):
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_heads: int = 4, output_dim: int = 3, dropout: float = 0.2):
        super().__init__(input_dim, output_dim)
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        
        x = x.unsqueeze(1)
        
        attn_out, _ = self.attention(x, x, x)
        attn_out = attn_out.squeeze(1)
        
        x = self.mlp(attn_out)
        
        return self.output_layer(x)
