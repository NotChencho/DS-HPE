"""
PyTorch model architectures for energy consumption prediction.
"""
import torch
import torch.nn as nn


class BaseNeuralNetwork(nn.Module):
    """Base class for neural network models."""
    
    def __init__(self, input_dim: int, output_dim: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim


class SimpleMLP(BaseNeuralNetwork):
    """Simple Multi-Layer Perceptron for regression."""
    
    def __init__(self, input_dim: int, hidden_dims: list = None, 
                 output_dim: int = 3, dropout: float = 0.3):
        super().__init__(input_dim, output_dim)
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
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


class AttentionMLP(BaseNeuralNetwork):
    """MLP with attention mechanism for energy consumption prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 144, 
                 num_heads: int = 6, output_dim: int = 3, dropout: float = 0.5):
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
        # Embed input
        x = self.embedding(x)
        
        # Add batch dimension for attention if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        
        # Remove the sequence dimension
        attn_output = attn_output.squeeze(1)
        
        # MLP
        x = self.mlp(attn_output)
        
        # Output
        return self.output_layer(x)


class RegressionMLP(BaseNeuralNetwork):
    """Multi-output regression MLP."""
    
    def __init__(self, input_dim: int, hidden_dims: list = None, 
                 output_dim: int = 3, dropout: float = 0.2):
        super().__init__(input_dim, output_dim)
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class DenseNN(nn.Module):
    """
    Dense Neural Network with configurable hidden layers
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], output_dim=3, dropout=0.3):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer sizes
            output_dim: Number of output targets (3 for node, mem, cpu)
            dropout: Dropout probability
        """
        super(DenseNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)