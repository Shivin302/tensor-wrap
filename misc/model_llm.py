"""
PyTorch implementation of a simple transformer-based language model.
"""

import torch
import torch.nn as nn

class TransformerLLM(nn.Module):
    """A simple transformer-based language model."""
    
    def __init__(self, vocab_size=50000, d_model=768, nhead=12, num_layers=12):
        super().__init__()
        self.model_type = "llm"
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # Input shape: [batch_size, seq_len]
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.lm_head(x)
        return x

def create_model():
    """Creates and returns a transformer-based language model."""
    return TransformerLLM()
