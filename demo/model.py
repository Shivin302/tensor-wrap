import torch
import torch.nn as nn
import math


vocab_size = 1000
d_model = 64
nhead = 4
d_ff = 128
max_seq_len = 50
batch_inputs = torch.randint(0, vocab_size, (32, 20))

class TransformerLLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, d_ff, max_seq_len):
        super(TransformerLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.transformer_layer = nn.TransformerEncoderLayer(d_model, nhead, d_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        batch_size, seq_len = x.size()
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.encoder(x)
        x = self.fc(x)
        return x



def run_TransformerLLM():
    model = TransformerLLM(vocab_size, d_model, nhead, d_ff, max_seq_len)
